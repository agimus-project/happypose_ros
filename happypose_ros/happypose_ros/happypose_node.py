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
from rclpy.qos import qos_profile_system_default
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos_overriding_options import QoSOverridingOptions

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
    get_object_symmetries_msg,
)

from happypose_msgs.msg import ObjectSymmetriesArray  # noqa: E402

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros  # noqa: E402


def happypose_worker_proc(
    worker_free: mp.Value,
    observation_tensor_queue: mp.Queue,
    results_queue: mp.Queue,
    symmetries_queue: mp.Queue,
    params_queue: mp.Queue,
) -> None:
    """Function used to trigger worker process.

    :param worker_free: Boolean, shared value indicating if a worker is free to start processing new data.
    :type worker_free: multiprocessing.Value
    :param observation_tensor_queue: Queue used to pass images from the main process to worker process.
    :type observation_tensor_queue: multiprocessing.Queue
    :param result_queue: Queue used to pass dict with the results to from worker process to the main process.
    :type result_queue: multiprocessing.Queue
    :param params_queue: Queue used to pass new incoming ROS parameters in a form of a dict.
    :type params_queue: multiprocessing.Queue
    """
    # Initialize the pipeline
    pipeline = HappyPosePipeline(params_queue.get())
    # Inform ROS node about the dataset
    symmetries_queue.put(pipeline.get_dataset())

    # Notify parent that initialization has finished
    with worker_free.get_lock():
        worker_free.value = True

    try:
        while True:
            # Await any data on all the input queues
            observation = observation_tensor_queue.get(block=True, timeout=None)
            if observation is None:
                break

            # Update inference args if available
            try:
                params = params_queue.get_nowait()
                pipeline.update_params(params)
                symmetries_queue.put(pipeline.get_dataset())
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
    """Main class wrapping HappyPose into ROS 2 node."""

    def __init__(self, node_name: str = "happypose_node", **kwargs) -> None:
        """Initializes the HappyPoseNode object. Validates ROS parameters and creates
        need subscribers and publishers. Initializes worker thread.

        :param node_name: Name of the created ROS node, defaults to "happypose_node"
        :type node_name: str, optional
        :raises Exception: Initialization of the generate_parameter_library object failed.
        :raises rclpy.ParameterException: More cameras expected than provided.
        :raises rclpy.ParameterException: No leading camera was passed as a parameter.
        :raises rclpy.ParameterException: More than one leading camera was passed as parameter.
        :raises rclpy.ParameterException: Leading camera has TF parameter enabled.
        """
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
        self._symmetries_queue = ctx.Queue(1)
        self._params_queue = ctx.Queue(1)

        self._update_dynamic_params(True)

        # TODO check efficiency of a single queue of ObservationTensors
        self._happypose_worker = ctx.Process(
            target=happypose_worker_proc,
            name="happypose_worker",
            args=(
                self._worker_free,
                self._observation_tensor_queue,
                self._results_queue,
                self._symmetries_queue,
                self._params_queue,
            ),
        )
        self._await_results_task = None

        # TODO once MegaPose is available initialization
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
            name: CameraWrapper(
                self,
                self._params.cameras,
                name,
                self._on_image_cb,
                self._params.use_depth,
            )
            for name in self._params.camera_names
        }

        self._last_pipeline_trigger = self.get_clock().now()

        self._detections_publisher = self.create_publisher(
            Detection2DArray,
            "happypose/detections",
            qos_profile=qos_profile_system_default,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )
        # TODO remove. This is debug
        self._detections_cosypose_publisher = self.create_publisher(
            Detection2DArray,
            "cosypose/detections",
            qos_profile=qos_profile_system_default,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )
        self._vision_info_publisher = self.create_publisher(
            VisionInfo,
            "happypose/vision_info",
            qos_profile=qos_profile_system_default,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )
        # TODO remove. This is debug
        self._vision_info_cosypose_publisher = self.create_publisher(
            VisionInfo,
            "cosypose/vision_info",
            qos_profile=qos_profile_system_default,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        self._symmetries_publisher = self.create_publisher(
            ObjectSymmetriesArray,
            "happypose/object_symmetries",
            # Set the message to be "latched"
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        if (
            self._params.pose_estimator_type == "cosypose"
            and self._params.use_depth
            and self._params.cosypose.renderer.renderer_type != "panda3d"
        ):
            e = ParameterException(
                "Use of any other renderer than `panda3d` is not supported when "
                + "depth refinement is enabled!",
                ("pose_estimator_type", "use_depth", "cosypose.renderer"),
            )
            self.get_logger().error(str(e))
            raise e

        compressed_cam = next(
            (
                name
                for name in self._cameras.keys()
                if self._params.cameras.get_entry(name).compressed
            ),
            None,
        )
        if self._params.use_depth and compressed_cam is not None:
            e = ParameterException(
                "Use of depth pose refinement with compressed images is not supported!",
                ("use_depth", f"cameras.{compressed_cam}.compressed"),
            )
            self.get_logger().error(str(e))
            raise e

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

        leading_cam_params = self._params.cameras.get_entry(self._leading_camera)

        if leading_cam_params.publish_tf:
            e = ParameterException(
                "Leading camera can not publish TF",
                (
                    f"cameras.{self._leading_camera}.publish_tf",
                    f"cameras.{self._leading_camera}.leading",
                ),
            )
            self.get_logger().error(str(e))
            raise e

        if leading_cam_params.estimated_tf_frame_id != "":
            e = ParameterException(
                "Leading camera can not have `frame_id` overwritten",
                (
                    f"cameras.{self._leading_camera}.estimated_tf_frame_id",
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
                MarkerArray,
                "happypose/markers",
                qos_profile=qos_profile_system_default,
                qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            )

        # Start the worker when all possible errors are handled
        self._happypose_worker.start()

        # Start spinner waiting for updates in symmetries
        self._symmetries_queue_task = Thread(target=self._symmetries_queue_spinner)
        self._symmetries_queue_task.start()

        self.get_logger().info(
            "Node initialized. Waiting for HappyPose worker to initialize...",
        )

    def destroy_node(self) -> None:
        """Destroys the node and closes all queues."""
        # Signal closing of queues
        if self._symmetries_queue is not None:
            self._symmetries_queue.put(None)
        if self._observation_tensor_queue is not None:
            self._observation_tensor_queue.put(None)

        # Close receiving queue
        if self._results_queue is not None:
            self._results_queue.close()

        # Stop threads
        if self._await_results_task is not None:
            self._await_results_task.join()
        if self._symmetries_queue_task is not None:
            self._symmetries_queue_task.join()
            if self._symmetries_queue is not None:
                self._symmetries_queue.close()

        # Stop worker process
        if self._happypose_worker is not None:
            self._happypose_worker.join()
            self._happypose_worker.terminate()
        if self._observation_tensor_queue is not None:
            self._observation_tensor_queue.close()
        super().destroy_node()

    def _update_dynamic_params(self, on_init: bool = False) -> None:
        """Updates ROS parameters and passes parsed parameters to a worker process via queue.

        :param on_init: Whether to skip steps when node is initialized, defaults to False
        :type on_init: bool, optional
        """
        self._param_listener.refresh_dynamic_parameters()
        self._params = self._param_listener.get_params()
        self._stamp_select_strategy = {"newest": max, "oldest": min, "average": mean}[
            self._params.time_stamp_strategy
        ]
        # Update internal params of cameras
        if not on_init:
            for cam in self._cameras.values():
                cam.update_params(self._params.cameras)

        # Clear the queue from old data
        while not self._params_queue.empty():
            self._params_queue.get()
        # Put new data to the queue
        self._params_queue.put(params_to_dict(self._params))
        if self._params.verbose_info_logs and not on_init:
            self.get_logger().info("Parameter change occurred.")

    def _symmetries_queue_spinner(self) -> None:
        """Awaits new data in a queue with symmetries and publishes them on a ROS topic"""
        try:
            while True:
                symmetries = self._symmetries_queue.get(block=True, timeout=None)
                if symmetries is None:
                    break

                header = Header(
                    frame_id="",
                    stamp=self.get_clock().now().to_msg(),
                )
                symmetries_msg = get_object_symmetries_msg(symmetries, header)
                self._symmetries_publisher.publish(symmetries_msg)
        # Queue is closed
        except ValueError:
            pass

    def _on_image_cb(self) -> None:
        """Callback function used to synchronize incoming images from different cameras.
        Performs basic checks if worker process is available and can start HappyPose pipeline.
        """
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
        """Triggers the HappyPose pipeline. Checks if camera images were received
        within an acceptable time frame. Creates Observation tensor and passes it to the
        worker process. Spawns thread to asynchronously await results.
        """
        now = self.get_clock().now()
        # If timeout at 0.0 accept all images
        skip_timeout = math.isclose(self._params.cameras.timeout, 0.0)
        processed_cameras = dict(
            filter(
                lambda cam: (
                    cam[1].ready()
                    and (
                        skip_timeout
                        # Fallback to checking the timestamp if not skipped
                        or (now - cam[1].get_last_image_stamp())
                        < Duration(seconds=self._params.cameras.timeout)
                    )
                ),
                self._cameras.items(),
            )
        )

        if self._leading_camera not in processed_cameras.keys():
            if (now - self._last_pipeline_trigger) > Duration(seconds=20):
                self.get_logger().warn(
                    "Failed to include leading camera in"
                    " the pipeline for past 20 seconds.",
                    throttle_duration_sec=20.0,
                )
            return

        leading_cam_shape = processed_cameras[
            self._leading_camera
        ].get_last_image_shape()

        def __check_shape_and_log(name: str, cam: CameraWrapper) -> bool:
            image_shape = cam.get_last_image_shape()
            if image_shape != leading_cam_shape:
                self.get_logger().warn(
                    f"Mismatch in image shapes for camera '{name}' and leading camera!"
                    f" Has shape '{image_shape}', while expected '{leading_cam_shape}'!"
                    f" Image from camera '{name}' will be discarded!",
                    throttle_duration_sec=5.0,
                )
                return False
            return True

        processed_cameras = {
            name: cam
            for name, cam in processed_cameras.items()
            if __check_shape_and_log(name, cam)
        }

        if len(processed_cameras) < self._params.cameras.n_min_cameras:
            # Throttle logs to either 5 times the timeout or 10 seconds
            timeout = max(5.0 * self._params.cameras.timeout, 10.0)
            if (now - self._last_pipeline_trigger) > (Duration(seconds=timeout)):
                self.get_logger().warn(
                    "Unable to start pipeline! Not enough camera"
                    " views before timeout reached!",
                    throttle_duration_sec=timeout,
                )
            return

        rgb_tensor = torch.as_tensor(
            np.stack([cam.get_last_rgb_image() for cam in processed_cameras.values()])
        ).permute(0, 3, 1, 2)

        if self._params.use_depth:
            depth_tensor = torch.as_tensor(
                np.stack(
                    [cam.get_last_depth_image() for cam in processed_cameras.values()]
                )
            ).unsqueeze(1)
        else:
            depth_tensor = None

        K_tensor = torch.as_tensor(
            np.stack([cam.get_last_k_matrix() for cam in processed_cameras.values()])
        )

        # Enable shared memory to increase performance
        rgb_tensor.to(self._device).share_memory_()
        K_tensor.to(self._device).share_memory_()
        if self._params.use_depth:
            depth_tensor.to(self._device).share_memory_()

        observation = ObservationTensor.from_torch_batched(
            rgb=rgb_tensor, depth=depth_tensor, K=K_tensor
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
        """Awaits results from the worker process. Converts the results into ROS messages
        and publishes them.

        :param cam_data: Camera names with associated frame id and time stamps.
        :type cam_data: dict
        """
        try:
            # Await any data on all the input queues
            if self._params.verbose_info_logs:
                self.get_logger().info("Awaiting results...")

            results = self._results_queue.get(block=True, timeout=None)

            header = Header(
                frame_id=cam_data[self._leading_camera]["frame_id"],
                # Choose sorted time stamp
                stamp=Time(
                    nanoseconds=self._stamp_select_strategy(
                        [cam["stamp"].nanoseconds for cam in cam_data.values()]
                    )
                ).to_msg(),
            )

            if results is not None:
                if self._params.verbose_info_logs:
                    self.get_logger().info(f"Detected {len(results['infos'])} objects.")
                    rounded_timings = {
                        k: round(v, 4) for k, v in results["timings"].items()
                    }
                    self.get_logger().info(f"Timings {rounded_timings}")

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
                    leading_cam_pose_inv = results["camera_poses"][
                        lead_cam_idx
                    ].inverse()
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

                # In case of multi-view, do not use bounding boxes
                detections = get_detection_array_msg(
                    results, header, has_bbox=not self._multiview
                )

                # TODO remove. This is debug
                cosypose_detections = get_detection_array_msg(
                    results,
                    header,
                    has_bbox=not self._multiview,
                    result="cosypose_poses",
                )

            else:
                if self._params.verbose_info_logs:
                    self.get_logger().info("No objects detected.")

                detections = Detection2DArray(
                    header=header,
                    detections=[],
                )

                # TODO remove. This is debug
                cosypose_detections = Detection2DArray(
                    header=header,
                    detections=[],
                )

            self._detections_publisher.publish(detections)
            # TODO remove. This is debug
            self._detections_cosypose_publisher.publish(cosypose_detections)

            self._vision_info_msg.header = header
            self._vision_info_publisher.publish(self._vision_info_msg)
            self._vision_info_cosypose_publisher.publish(self._vision_info_msg)

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
    """Creates the ROS node object and spins it."""
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
