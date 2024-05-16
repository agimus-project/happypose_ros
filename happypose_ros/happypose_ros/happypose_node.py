from ctypes import c_bool
import math
import numpy as np
from statistics import mean
from threading import Thread
import torch
import torch.multiprocessing as mp
import queue


import rclpy
from rclpy.duration import Duration
from rclpy.exceptions import ParameterException
import rclpy.logging
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import TransformBroadcaster

from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection2DArray, VisionInfo

from happypose.toolbox.datasets.datasets_cfg import make_object_dataset
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.logging import get_logger

logger = get_logger(__name__)

from happypose_ros.camera_wrapper import CameraWrapper  # noqa: E402
from happypose_ros.inference_pipeline import HappyPosePipeline  # noqa: E402
from happypose_ros.utils import (  # noqa: E402
    params_to_dict,
    get_camera_transform,
    get_detection_array_msg,
    get_marker_array_msg,
)

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros  # noqa: E402


def happypose_worker_proc(
    params: dict,
    worker_free: mp.Value,
    observation_tensor_queue: mp.Queue,
    results_queue: mp.Queue,
    params_queue: mp.Queue,
) -> None:
    # Initialize the pipeline
    pipeline = HappyPosePipeline(params)

    # Notify parent that initialization has finished
    with worker_free.get_lock():
        worker_free.value = True

    try:
        while True:
            # Await any data on all the input queues
            observation = observation_tensor_queue.get(block=True, timeout=None)

            # Update inference args if available
            try:
                params = params_queue.get_nowait()
                pipeline.update_params(params)
            except queue.Empty:
                pass

            results = pipeline(observation)
            results_queue.put(results)

            # Notify parent that processing finished
            with worker_free.get_lock():
                worker_free.value = True
    # Queues are closed or SIGINT received
    except (ValueError, KeyboardInterrupt):
        pass
    except Exception as e:
        logger.error(f"Worker got exception: {str(e)}. Exception type: {type(e)}.")

    logger.info("HappyPoseWorker finished job.")


class HappyPoseNode(Node):
    def __init__(self, node_name: str = "happypose_node", **kwargs) -> None:
        super().__init__(node_name, **kwargs)

        try:
            self._param_listener = happypose_ros.ParamListener(self)
            self._params = self._param_listener.get_params()
        except Exception as e:
            self.get_logger().error(str(e))
            raise e

        self._device = self._params.device

        ctx = mp.get_context("spawn")
        self._worker_free = ctx.Value(c_bool, False)
        self._observation_tensor_queue = ctx.Queue(1)
        self._results_queue = ctx.Queue(1)
        self._params_queue = ctx.Queue(1)

        # TODO check efficiency of a single queue of ObservationTensors
        self._happypose_worker = ctx.Process(
            target=happypose_worker_proc,
            name="happypose_worker",
            args=(
                params_to_dict(self._params),
                self._worker_free,
                self._observation_tensor_queue,
                self._results_queue,
                self._params_queue,
            ),
        )
        self._await_results_task = None

        # TODO once Megapose is available initialization
        # should be handled better
        self._vision_info_msg = VisionInfo(
            method=self._params.pose_estimator_type,
            # TODO set this parameter to something more meaningful
            database_location=make_object_dataset(
                self._params.cosypose.dataset_name
            ).ds_dir.as_posix(),
            database_version=0,
        )

        # Each camera registers its topics and fires a synchronization callback on new image
        self._cameras = {
            name: CameraWrapper(self, self._params.cameras, name, self._on_image_cb)
            for name in self._params.camera_names
        }

        self._last_pipeline_trigger = self.get_clock().now()

        self._detections_publisher = self.create_publisher(
            Detection2DArray, "happypose/detections", 10
        )
        self._vision_info_publisher = self.create_publisher(
            VisionInfo, "happypose/vision_info", 10
        )

        if self._params.cameras.n_min_cameras > len(self._cameras):
            e = ParameterException(
                "Minimum number of cameras to trigger the pipeline is"
                " greater than the number of available cameras",
                ("cameras.n_min_cameras", "cameras"),
            )
            self.get_logger().error(str(e))
            raise e

        leading_cams = sum(
            [
                self._params.cameras.get_entry(name).leading
                for name in self._cameras.keys()
            ]
        )
        if leading_cams == 0:
            e = ParameterException(
                "No leading cameras specified. Modify parameter",
                ("cameras.<camera_name>.leading"),
            )
            self.get_logger().error(str(e))
            raise e

        elif leading_cams > 1:
            e = ParameterException(
                "HappyPose can use only single leading camera",
                [
                    f"cameras.{name}.leading"
                    for name in self._cameras.keys()
                    if self._params.cameras.get_entry(name).leading
                ],
            )
            self.get_logger().error(str(e))
            raise e

        # Find leading camera name
        self._leading_camera = next(
            filter(
                lambda name: self._params.cameras.get_entry(name).leading, self._cameras
            )
        )

        if self._params.cameras.get_entry(self._leading_camera).publish_tf:
            e = ParameterException(
                "Leading camera can not publish TF",
                (
                    f"cameras.{self._leading_camera}.publish_tf",
                    f"cameras.{self._leading_camera}.leading",
                ),
            )
            self.get_logger().error(str(e))
            raise e

        self._multiview = len(self._cameras) > 1
        if self._multiview:
            self.get_logger().info(
                "Node configured to run in multi-view mode."
                " Minimum number of views expected: "
                f"{self._params.cameras.n_min_cameras}."
            )
            # Broadcast TF if any camera needs it
            if any(
                [
                    self._params.cameras.get_entry(name).publish_tf
                    for name in self._cameras
                ]
            ):
                self._tf_broadcaster = TransformBroadcaster(self)

        # Create debug publisher
        if self._params.visualization.publish_markers:
            self._marker_publisher = self.create_publisher(
                MarkerArray, "happypose/markers", 10
            )

        # Start the worker when all possible errors are handled
        self._update_dynamic_params(True)
        self._happypose_worker.start()

        self.get_logger().info(
            "Node initialized. Waiting for HappyPose worker to initialize...",
        )

    def destroy_node(self) -> None:
        if self._observation_tensor_queue is not None:
            self._observation_tensor_queue.close()
        if self._results_queue is not None:
            self._results_queue.close()
        if self._await_results_task is not None:
            self._await_results_task.join()
        if self._happypose_worker is not None:
            self._happypose_worker.join()
            self._happypose_worker.terminate()
        super().destroy_node()

    def _update_dynamic_params(self, on_init: bool = False) -> None:
        self._param_listener.refresh_dynamic_parameters()
        self._params = self._param_listener.get_params()
        self._stamp_select_strategy = {"newest": max, "oldest": min, "average": mean}[
            self._params.time_stamp_strategy
        ]
        # Clear the queue from old data
        while not self._params_queue.empty():
            self._params_queue.get()
        # Put new data to the queue
        self._params_queue.put(params_to_dict(self._params))
        if self._params.verbose_info_logs and not on_init:
            self.get_logger().info("Parameter change occurred.")

    def _on_image_cb(self) -> None:
        #  Check if parameters didn't change
        if self._param_listener.is_old(self._params):
            self._update_dynamic_params()

        # Skip if task was initialized and is still running
        if self._await_results_task and self._await_results_task.is_alive():
            return

        # Skip if worker is still processing the data
        with self._worker_free.get_lock():
            if not self._worker_free.value:
                return

        # Print this log message only once in the beginning
        self.get_logger().info(
            "HappyPose initialized. Starting to process incoming images.", once=True
        )

        self._trigger_pipeline()

    def _trigger_pipeline(self) -> None:
        now = self.get_clock().now()
        # If timeout at 0.0 accept all images
        skipp_timeout = math.isclose(self._params.cameras.timeout, 0.0)
        processed_cameras = dict(
            filter(
                lambda cam: (
                    cam[1].ready()
                    and (
                        skipp_timeout
                        # Fallback to checking the timestamp if not skipped
                        or (now - cam[1].get_last_image_stamp())
                        < Duration(seconds=self._params.cameras.timeout)
                    )
                ),
                self._cameras.items(),
            )
        )

        if len(processed_cameras) < self._params.cameras.n_min_cameras:
            # Throttle logs to either 5 times the timeout or 10 seconds
            timeout = max(5 * self._params.cameras.timeout, 10.0)
            if (now - self._last_pipeline_trigger) > (Duration(seconds=timeout)):
                self.get_logger().warn(
                    "Unable to start pipeline! Not enough camera"
                    " views before timeout reached!",
                    throttle_duration_sec=timeout,
                )
            return

        if self._leading_camera not in processed_cameras.keys():
            if (now - self._last_pipeline_trigger) > Duration(seconds=20):
                self.get_logger().warn(
                    "Failed to include leading camera in"
                    " the pipeline for past 20 seconds.",
                    throttle_duration_sec=20.0,
                )
            return

        # TODO implement depth info
        rgb_tensor = torch.as_tensor(
            np.stack([cam.get_last_rgb_image() for cam in processed_cameras.values()])
        )
        K_tensor = torch.as_tensor(
            np.stack([cam.get_last_k_matrix() for cam in processed_cameras.values()])
        )
        if rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)

        # Enable shared memory to increase performance
        rgb_tensor.to(self._device).share_memory_()
        K_tensor.to(self._device).share_memory_()

        observation = ObservationTensor.from_torch_batched(
            rgb=rgb_tensor, depth=None, K=K_tensor
        )
        observation.to(self._device)
        self._observation_tensor_queue.put(observation)

        with self._worker_free.get_lock():
            self._worker_free.value = False
        self._last_pipeline_trigger = now

        # As of python 3.7 dict is insertion ordered so this can be
        # used to unwrap tensors later when using multi-view
        cam_data = {
            name: {
                "frame_id": cam.get_last_image_frame_id(),
                "stamp": cam.get_last_image_stamp(),
            }
            for name, cam in processed_cameras.items()
        }

        self.get_logger().info(
            "First inference might take longer, as the pipeline is still loading.",
            once=True,
        )

        self._await_results_task = Thread(target=self._await_results, args=(cam_data,))
        self._await_results_task.start()

    def _await_results(self, cam_data: dict) -> None:
        try:
            # Await any data on all the input queues
            if self._params.verbose_info_logs:
                self.get_logger().info("Awaiting results...")

            results = self._results_queue.get(block=True, timeout=None)

            if results is None:
                if self._params.verbose_info_logs:
                    self.get_logger().info("No objects detected.")
                return

            if self._params.verbose_info_logs:
                self.get_logger().info(f"Detected {len(results['infos'])} objects.")

            if self._multiview:
                missing_cameras = len(cam_data) - len(results["camera_infos"])
                if missing_cameras > 0:
                    # Keep only the cameras that were not discarded in multiview
                    cam_data = {
                        name: cam
                        for i, (name, cam) in enumerate(cam_data.items())
                        if i in results["camera_infos"].view_id.values
                    }

                    if self._leading_camera not in cam_data.keys():
                        self.get_logger().error(
                            f"Leading camera '{self._leading_camera}'"
                            " was discarded when performing multi-view!"
                        )
                        return

                    if self._params.verbose_info_logs:
                        self.get_logger().warn(
                            f"{missing_cameras} camera views were discarded in multi-view."
                        )

                lead_cam_idx = list(cam_data.keys()).index(self._leading_camera)
                # Transform objects into leading camera's reference frame
                leading_cam_pose_inv = results["camera_poses"][lead_cam_idx].inverse()
                # Transform detected objects' poses
                results["poses"] = leading_cam_pose_inv @ results["poses"]
                # Create mask of cameras that expect to have TF published
                cams_to_tf = [
                    self._params.cameras.get_entry(name).publish_tf
                    for name in cam_data.keys()
                ]
                # Transform cameras' poses
                results["camera_poses"][cams_to_tf] = (
                    leading_cam_pose_inv @ results["camera_poses"][cams_to_tf]
                )

            header = Header(
                frame_id=cam_data[self._leading_camera]["frame_id"],
                # Choose sorted time stamp
                stamp=Time(
                    nanoseconds=self._stamp_select_strategy(
                        [cam["stamp"].nanoseconds for cam in cam_data.values()]
                    )
                ).to_msg(),
            )

            # In case of multi-view, do not use bounding boxes
            detections = get_detection_array_msg(
                results, header, has_bbox=not self._multiview
            )
            self._detections_publisher.publish(detections)

            self._vision_info_msg.header = header
            self._vision_info_publisher.publish(self._vision_info_msg)

            if self._params.visualization.publish_markers:
                # TODO consider better path handling
                markers = get_marker_array_msg(
                    detections,
                    f"file://{self._vision_info_msg.database_location}",
                    prefix=self._params.cosypose.dataset_name + "-",
                    dynamic_opacity=self._params.visualization.markers.dynamic_opacity,
                    marker_lifetime=self._params.visualization.markers.lifetime,
                )
                self._marker_publisher.publish(markers)

            if self._multiview:
                # Assume HappyPose is already returning cameras in the same order it received them
                for pose, (name, cam) in zip(results["camera_poses"], cam_data.items()):
                    if (
                        not self._params.cameras.get_entry(name).leading
                        and self._params.cameras.get_entry(name).publish_tf
                    ):
                        self._tf_broadcaster.sendTransform(
                            get_camera_transform(pose, header, cam["frame_id"])
                        )
        # Queue was closed
        except ValueError:
            return
        except Exception as e:
            self.get_logger().error(f"Publishing data failed. Reason: {str(e)}")


def main() -> None:
    rclpy.init()
    try:
        happypose_node = HappyPoseNode()
        rclpy.spin(happypose_node)
        happypose_node.destroy_node()
    except (KeyboardInterrupt, ParameterException):
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
