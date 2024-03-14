from ctypes import c_bool
import numpy as np

import torch
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node

from happypose.toolbox.inference.types import ObservationTensor

from happypose_ros.camera_wrapper import CameraWrapper
from happypose_ros.inference_pipeline import HappyposePipeline
from happypose_ros.happypose_ros_parameters import happypose_ros

from happypose.toolbox.utils.logging import get_logger

logger = get_logger(__name__)


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
        self._device = params.device
        self._worker_free = worker_flag
        self._stop_worker = stop_worker
        self._image_queue = image_queue
        self._depth_queue = depth_queue
        self._k_queue = k_queue
        self._result_queue = result_queue

        # Initialize the pipeline
        self._pipeline = HappyposePipeline(params)

        # Notify parent that initialization has finished
        with self._worker_free.get_lock():
            self._worker_free.value = True

    def run(self) -> None:
        # try:
        while True:
            # Stop the process if paren is stopped
            with self._stop_worker.get_lock():
                if self._stop_worker.value:
                    logger.error("Worker finishing job")
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
            self._output_queue.put(result)

            logger.error("result in queue")

            # Notify parent that processing finished
            with self._worker_free.get_lock():
                self._worker_free.value = True

            logger.error("worker free")
        # except Exception as e:
        #     logger.error(f"Worker got exception: {str(e)}")
        #     return


class HappyposeNode(Node):
    def __init__(self) -> None:
        super().__init__("happypose_node")

        self._param_listener = happypose_ros.ParamListener(self)
        self._params = self._param_listener.get_params()

        self._device = self._params.device

        # self._worker_free = mp.Value(c_bool, False)
        self._worker_free = mp.Value(c_bool, True)
        self._stop_worker = mp.Value(c_bool, False)
        self._image_queue = mp.Queue(1)
        self._depth_queue = mp.Queue(1)
        self._k_queue = mp.Queue(1)
        self._result_queue = mp.Queue(1)

        self._happypose_worker = HappyposeWorker(
            self._params,
            self._worker_free,
            self._stop_worker,
            self._image_queue,
            self._depth_queue,
            self._k_queue,
            self._result_queue,
        )
        self._happypose_worker.start()

        # Each camera registers it's topics and fires synchronisation callback on new image
        self._cameras = {
            name: CameraWrapper(self, self._params.cameras, name, self._on_image_cb)
            for name in self._params.cameras.names
        }
        self._processed_cameras = []
        self._last_pipeline_trigger = None

        self.get_logger().info(
            "Node initialized. Waiting for Happypose to initialized...",
        )

    def destroy_node(self) -> None:
        with self._stop_worker.get_lock():
            self._stop_worker.value = True
        self._image_queue.close()
        self._depth_queue_queue.close()
        self._k_queue.close()
        self._happypose_worker.join()
        super().destroy_node()

    def _on_image_cb(self) -> None:
        # Skipp if executor is not initialized
        with self._worker_free.get_lock():
            if not self._worker_free.value:
                return

        # Print this log message only once in the beginnging
        self.get_logger().info(
            "Happypose initialized. Starting to process incomming images.", once=True
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
                # TODO Consider more meaningfull message
                self.get_logger().warn(
                    "Unable to start pipeline! Not enough camera views before timeout reached!",
                    throttle_duration_sec=5.0,
                )
            return

        # TODO propperly implement multiview
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


def main() -> None:
    rclpy.init()
    happypose_node = HappyposeNode()
    rclpy.spin(happypose_node)
    happypose_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
