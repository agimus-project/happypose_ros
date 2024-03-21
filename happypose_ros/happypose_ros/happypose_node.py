from ctypes import c_bool
from threading import Thread
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection2DArray, VisionInfo

from happypose.toolbox.datasets.datasets_cfg import make_object_dataset
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.logging import get_logger

logger = get_logger(__name__)

from happypose_ros.camera_wrapper import CameraWrapper  # noqa: E402
from happypose_ros.inference_pipeline import HappyposePipeline  # noqa: E402
from happypose_ros.utils import (  # noqa: E402
    params_to_dict,
    get_detection_array_msg,
    get_marker_array_msg,
)

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros  # noqa: E402


def happypose_worker_proc(
    params: happypose_ros.Params,
    worker_free: mp.Value,
    observation_tensor_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    # Initialize the pipeline
    pipeline = HappyposePipeline(params)

    # Notify parent that initialization has finished
    with worker_free.get_lock():
        worker_free.value = True

    try:
        while True:
            # Await any data on all the input queues
            observation = observation_tensor_queue.get(block=True, timeout=None)

            result = pipeline(observation)
            result_queue.put(result)

            # Notify parent that processing finished
            with worker_free.get_lock():
                worker_free.value = True
    # Queues are closed or SIGINT recieved
    except (ValueError, KeyboardInterrupt):
        pass
    except Exception as e:
        logger.error(f"Worker got exception: {str(e)}. Exception type: {type(e)}.")

    logger.error("HappyposeWorker finished job.")


class HappyposeNode(Node):
    def __init__(self) -> None:
        super().__init__("happypose_node")

        self._param_listener = happypose_ros.ParamListener(self)
        self._params = self._param_listener.get_params()

        self._device = self._params.device

        ctx = mp.get_context("spawn")
        self._worker_free = ctx.Value(c_bool, False)
        self._observation_tensor_queue = ctx.Queue(1)
        self._result_queue = ctx.Queue(1)

        # TODO check efficiency of a single queue of ObservationTensors
        self._happypose_worker = ctx.Process(
            target=happypose_worker_proc,
            name="happypose_worker",
            args=(
                params_to_dict(self._params),
                self._worker_free,
                self._observation_tensor_queue,
                self._result_queue,
            ),
        )
        self._happypose_worker.start()

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
            for name in self._params.cameras.names
        }
        self._processed_cameras = []
        self._last_pipeline_trigger = None

        self._detections_publisher = self.create_publisher(
            Detection2DArray, "happypose/detections", 10
        )
        self._vision_info_publisher = self.create_publisher(
            VisionInfo, "happypose/vision_info", 10
        )

        # Create debug publisher
        if self._params.publish_markers:
            self._marker_publisher = self.create_publisher(
                MarkerArray, "happypose/markers", 10
            )

        self.get_logger().info(
            "Node initialized. Waiting for Happypose to initialized...",
        )

    def destroy_node(self):
        if self._observation_tensor_queue is not None:
            self._observation_tensor_queue.close()
        if self._result_queue is not None:
            self._result_queue.close()
        if self._await_results_task is not None:
            self._await_results_task.join()
        if self._happypose_worker is not None:
            self._happypose_worker.join()
            self._happypose_worker.terminate()
        super().destroy_node()

    def _on_image_cb(self) -> None:
        # Skip if task was initialized and is still running
        if self._await_results_task and self._await_results_task.is_alive():
            return

        # Skip if worker is still processing the data
        with self._worker_free.get_lock():
            if not self._worker_free.value:
                return

        # Print this log message only once in the beginning
        self.get_logger().info(
            "Happypose initialized. Starting to process incoming images.", once=True
        )

        self._trigger_pipeline()

    def _trigger_pipeline(self):
        self.get_logger().info(
            "First inference might take longer, as the pipeline is still loading.",
            once=True,
        )

        now = self.get_clock().now()
        if self._params.cameras.timeout:
            processed_cameras = [
                name
                for name, cam in self._cameras.items()
                if (now - cam.get_last_image_stamp()) > self._params.cameras.timeout
            ]
        else:
            processed_cameras = list(self._cameras.keys())

        if len(processed_cameras) < self._params.cameras.n_min_cameras:
            if self._last_pipeline_trigger and (now - self._last_pipeline_trigger) > (
                5 * self._params.cameras.timeout
            ):
                # TODO Consider more meaningful message
                self.get_logger().warn(
                    "Unable to start pipeline! Not enough camera "
                    + "views before timeout reached!",
                    throttle_duration_sec=5.0,
                )
            return

        # TODO properly implement multi-view
        # TODO implement depth info
        K, rgb = self._cameras[processed_cameras[0]].get_camera_data()
        observation = ObservationTensor.from_numpy(
            rgb=rgb, depth=None, K=K.reshape((3, 3))
        )
        observation.to(self._device)
        self._observation_tensor_queue.put(observation)

        with self._worker_free.get_lock():
            self._worker_free.value = False
        self._last_pipeline_trigger = now

        # As of python 3.7 dict is insertion ordered
        # so this can be used to unwrap tensors later
        # when using multi-view
        camera_inference_data = {
            cam: {
                "frame_id": self._cameras[cam].get_last_iamge_frame_id(),
                "stamp": self._cameras[cam].get_last_image_stamp(),
            }
            for cam in processed_cameras
        }

        self._await_results_task = Thread(
            target=self._await_results, args=(camera_inference_data,)
        )
        self._await_results_task.start()

    def _await_results(self, cam_data: dict) -> None:
        try:
            # Await any data on all the input queues
            if self._params.verbose_info_logs:
                self.get_logger().info("Awaiting results...")

            results = self._result_queue.get(block=True, timeout=None)

            if results is None:
                if self._params.verbose_info_logs:
                    self.get_logger().info("No objects detected.")
                return

            if self._params.verbose_info_logs:
                self.get_logger().info(f"Detected {len(results['infos'])} objects.")

            header = Header(
                # Use camera frame_id if single view
                frame_id=(
                    list(cam_data.values())[0]["frame_id"]
                    if len(cam_data) == 1
                    else self._params.frame_id
                ),
                # Use the oldest camera image time stamp
                stamp=min([cam["stamp"] for cam in cam_data.values()]).to_msg(),
            )

            detections = get_detection_array_msg(results, header)
            self._detections_publisher.publish(detections)

            self._vision_info_msg.header = header
            self._vision_info_publisher.publish(self._vision_info_msg)

            if self._params.publish_markers:
                # TODO consider better path handling
                markers = get_marker_array_msg(
                    detections,
                    f"file://{self._vision_info_msg.database_location}",
                    label_to_strip=self._params.cosypose.dataset_name + "-",
                    dynamic_opacity=True,
                )
                self._marker_publisher.publish(markers)
        # Queue was closed
        except ValueError:
            self.get_logger().error("queue closed")
            return
        except Exception as e:
            self.get_logger().error(f"Publishing data failed. Reason: {str(e)}")


def main() -> None:
    rclpy.init()
    happypose_node = HappyposeNode()
    try:
        rclpy.spin(happypose_node)
    except KeyboardInterrupt:
        pass
    happypose_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
