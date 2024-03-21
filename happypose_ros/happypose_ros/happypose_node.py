from ctypes import c_bool
import numpy as np
import torch
import torch.multiprocessing as mp

import rclpy
from rclpy.executors import MultiThreadedExecutor
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


class HappyposeWorker(mp.Process):
    def __init__(
        self,
        params: happypose_ros.Params,
        worker_flag: mp.Value,
        stop_worker: mp.Value,
        image_queue: mp.Queue,
        depth_queue: mp.Queue,
        k_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self._worker_free = worker_flag
        self._stop_worker = stop_worker
        self._image_queue = image_queue
        self._depth_queue = depth_queue
        self._k_queue = k_queue
        self._result_queue = result_queue

        torch.set_num_threads(1)

        # Initialize the pipeline
        self._pipeline = HappyposePipeline(params)

        # Notify parent that initialization has finished
        with self._worker_free.get_lock():
            self._worker_free.value = True

    def run(self) -> None:
        try:
            while True:
                # Stop the process if parent is stopped
                with self._stop_worker.get_lock():
                    if self._stop_worker.value:
                        logger.debug("Worker finishing job")
                        break

                # Await any data on all the input queues
                rgb_tensor = self._image_queue.get(block=True, timeout=None)
                # TODO implement depth
                # depth_tensor = self._depth_queue.get(block=True, timeout=None)
                K_tensor = self._k_queue.get(block=True, timeout=None)

                observation = ObservationTensor.from_torch_batched(
                    rgb=rgb_tensor, depth=None, K=K_tensor
                )

                result = self._pipeline(observation)
                self._result_queue.put(result)

                # Notify parent that processing finished
                with self._worker_free.get_lock():
                    self._worker_free.value = True

        except Exception as e:
            logger.error(f"Worker got exception: {str(e)}")
            return


class HappyposeNode(Node):
    def __init__(self) -> None:
        super().__init__("happypose_node")

        self._param_listener = happypose_ros.ParamListener(self)
        self._params = self._param_listener.get_params()

        self._device = self._params.device

        self._worker_free = mp.Value(c_bool, False)
        self._stop_worker = mp.Value(c_bool, False)
        self._image_queue = mp.Queue(1)
        self._depth_queue = mp.Queue(1)
        self._k_queue = mp.Queue(1)
        self._result_queue = mp.Queue(1)

        # TODO check efficiency of a single queue of ObservationTensors
        self._happypose_worker = HappyposeWorker(
            params_to_dict(self._params),
            self._worker_free,
            self._stop_worker,
            self._image_queue,
            self._depth_queue,
            self._k_queue,
            self._result_queue,
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
        self._camera_inference_data = {}
        self._last_pipeline_trigger = None

        self.get_logger().info(
            "Node initialized. Waiting for Happypose to initialized...",
        )

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

    def destroy_node(self) -> None:
        with self._stop_worker.get_lock():
            self._stop_worker.value = True
        self._image_queue.close()
        self._depth_queue.close()
        self._k_queue.close()
        self._result_queue.close()
        self._happypose_worker.join()
        super().destroy_node()

    def _on_image_cb(self) -> None:
        # Skip if task was initialized and is still running
        if self._await_results_task and not self._await_results_task.done():
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

        # As of python 3.7 dict is insertion ordered
        # so this can be used to unwrap tensors later
        self._camera_inference_data = {
            cam: {
                "frame_id": self._cameras[cam].get_last_iamge_frame_id(),
                "stamp": self._cameras[cam].get_last_image_stamp(),
            }
            for cam in processed_cameras
        }

        # TODO properly implement multi-view
        # TODO implement depth info
        K, rgb = self._cameras[processed_cameras[0]].get_camera_data()

        K_tensor = torch.as_tensor(np.array([K.reshape((3, 3))])).float()
        rgb_tensor = torch.as_tensor(np.array([rgb]))
        if rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)

        # Move tensors to the device and then allow shared memory
        rgb_tensor.to(self._device).share_memory_()
        K_tensor.to(self._device).share_memory_()

        self._image_queue.put(rgb_tensor)
        self._k_queue.put(K_tensor)

        with self._worker_free.get_lock():
            self._worker_free.value = False
        self._last_pipeline_trigger = now

        # Skip if task was initialized and it is still running
        if self._await_results_task and not self._await_results_task.done():
            raise RuntimeError(
                "Pose estimate task hasn't finished yet! Can't spawn new task!"
            )

        # Spawn task to await resulting data
        self._await_results_task = self.executor.create_task(self._await_results)

    def _await_results(self):
        try:
            # Await any data on all the input queues
            self.get_logger().info("Awaiting results...")
            results = self._result_queue.get(block=True, timeout=None)

            if not results:
                self.get_logger().info("No objects detected.")
                return

            self.get_logger().info(f"Detected {len(results['infos'])} objects.")

            cam_data = self._camera_inference_data
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
                # TODO better path handling
                markers = get_marker_array_msg(
                    detections,
                    f"file://{self._vision_info_msg.database_location}",
                    label_to_strip=self._params.cosypose.dataset_name + "-",
                    dynamic_opacity=True,
                )
                self._marker_publisher.publish(markers)
        except Exception as e:
            self.get_logger().error(f"Publishing data failed. Reason: {str(e)}")


def main() -> None:
    rclpy.init()
    happypose_node = HappyposeNode()

    executor = MultiThreadedExecutor()
    executor.add_node(happypose_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        happypose_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    torch.set_num_threads(1)
    mp.set_start_method("spawn", force=True)
    main()
