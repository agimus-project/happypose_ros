from ctypes import c_bool

import torch
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node

from happypose.toolbox.inference.types import ObservationTensor

from happypose_ros.camera_wrapper import CameraWrapper
from happypose_ros.pipeline import HappyposePipeline
from happypose_ros.happypose_ros_parameters import happypose_ros_parameters


class HappyposeWorker(mp.Process):
    def __init__(
        self,
        params: dict,
        worker_flag: mp.Value,
        stop_worker: mp.Value,
        image_queue: mp.Queue,
        depth_queue: mp.Queue,
        k_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self._device = params["device"]
        self._pipeline = HappyposePipeline(params, self._device)
        self._worker_free = worker_flag
        self._stop_worker = stop_worker
        self._image_queue = image_queue
        self._depth_queue = depth_queue
        self._k_queue = k_queue
        self._result_queue = result_queue

    def run(self) -> None:
        while True:
            # Stop the process if paren is stopped
            with self._stop_worker.get_lock():
                if self._stop_worker.vale:
                    break

            # Notify parent that processing is ongoing
            with self._stop_worker.get_lock():
                self._worker_free.vale = False

            # Await any data on all the input queues
            rgb_tensor = self._image_queue.get(block=True, timeout=None)
            # TODO implement depth
            # depth_tensor = self._depth_queue.get(block=True, timeout=None)
            K_tensor = self._k_queue.get(block=True, timeout=None)

            observation = ObservationTensor.from_torch_batched(
                rgb=rgb_tensor, K=K_tensor
            ).to(self._device)

            result = self._pipeline(observation)
            self._output_queue.put(result)

            # Notify parent that processing finished
            with self._stop_worker.get_lock():
                self._worker_free.vale = True


class HappyposeNode(Node):
    def __init__(self) -> None:
        super().__init__("happypose_node")

        self._param_listener = happypose_ros_parameters.ParamListener(self)
        self._params = self.param_listener.get_params()

        self._manager = mp.Manager()
        self._worker_free = self._manager.Value(c_bool, True)
        self._stop_worker = self._manager.Value(c_bool, False)
        self._image_queue = self._manager.Queue(1)
        self._depth_queue = self._manager.Queue(1)
        self._k_queue = self._manager.Queue(1)
        self._result_queue = self._manager.Queue(1)
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
            name: CameraWrapper(self, name, self._params, self._on_image_cb)
            for name in self._params.camera_names
        }
        self._processed_cameras = []
        self._last_pipeline_trigger = None

    def destroy_node(self) -> None:
        with self._stop_worker.get_lock():
            self._stop_worker.value = True
        self._image_queue.task_done()
        self._depth_queue_queue.task_done()
        self._k_queue.task_done()
        self._happypose_worker.join()
        super().destroy_node()

    def _on_image_cb(self) -> None:
        # Skipp if executor is not initialized
        with self._worker_free.get_lock():
            if not self._worker_free.value:
                return

        now = self.get_clock().now()
        if self._camera_timeout:
            processed_cameras = [
                name
                for name, cam in self._cameras.items()
                if (now - cam.get_last_image_stamp()) > self._camera_timeout
            ]
        else:
            processed_cameras = list(self._cameras.keys())

        if len(processed_cameras) < self._camera_min_num:
            if self._last_pipeline_trigger and (now - self._last_pipeline_trigger) > (
                5 * self._camera_timeout
            ):
                # TODO Consider more meaningfull message
                self.get_logger().warn(
                    "Unable to start pipeline! Not enough camera views before timeout reached!",
                    throttle_duration_sec=5.0,
                )
            return

        # TODO propperly implement multiview
        # TODO implement depth info
        K, rgb = processed_cameras[0].get_camera_data()
        rgb_tensor = torch.as_tensor(rgb).float() / 255.0
        K_tensor = torch.as_tensor(K).float()

        # Move tensors to the device and then allow shared memory
        rgb_tensor.to(self._device).share_memory_()
        K_tensor.to(self._device).share_memory_()
        self._image_queue.put(rgb_tensor)
        self._k_queue.put(K_tensor)

        self._last_pipeline_trigger = now


def main() -> None:
    rclpy.init()
    happypose_node = HappyposeNode()
    rclpy.spin(happypose_node)
    happypose_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
