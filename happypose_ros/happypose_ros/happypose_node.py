from typing import Union
from ctypes import c_bool

import torch
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node

from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.toolbox.inference.types import ObservationTensor

from happypose_ros.cosypose_loader import CosyPoseLoader
from happypose_ros.camera_wrapper import CameraWrapper


class HappyposeWorker(mp.Process):
    def __init__(
        self,
        device,
        pose_estimator: PoseEstimator,
        inference_params: dict[str, Union[float, list[str], int]],
        worker_flag: mp.Value,
        stop_worker: mp.Value,
        image_queue: mp.Queue,
        depth_queue: mp.Queue,
        k_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self._pose_estimator = pose_estimator
        self._inference_params = inference_params
        self._device = device
        self._worker_free = worker_flag
        self._stop_worker = stop_worker
        self._image_queue = image_queue
        self._depth_queue = depth_queue
        self._k_queue = k_queue
        self._result_queue = result_queue

    def run(self):
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

            preds, preds_extra = self._pose_estimator.run_inference_pipeline(
                observation=observation, run_detector=True, **self._inference_params
            )

            self._output_queue.put((preds, preds_extra))

            # Notify parent that processing finished
            with self._stop_worker.get_lock():
                self._worker_free.vale = True


class HappyposeNode(Node):
    def __init__(self) -> None:
        super().__init__("happypose_node")

        self.declare_parameter("device", "cpu")
        self.declare_parameter("model")

        cam_ns = "cameras"
        self.declare_parameter(cam_ns + "/timeout")
        self.declare_parameter(cam_ns + "/min_num")
        self.declare_parameter(cam_ns + "/cameras")

        self._device = self.get_parameter("device").get_parameter_value().string_value
        model_name = self.get_parameter("model").get_parameter_value().string_value
        if model_name == "cosypose":
            loader = CosyPoseLoader
        else:
            self.get_logger().error(
                f"Incorrect loader name: {model_name}! Only 'cosypose' is supported!"
            )
            rclpy.shutdown()

        pose_estimator = loader.load_pose_estimator(self, self._device)
        inference_params = loader.load_inference_params(self)

        self._manager = mp.Manager()
        self._worker_free = self._manager.Value(c_bool, True)
        self._stop_worker = self._manager.Value(c_bool, False)
        self._image_queue = self._manager.Queue(1)
        self._depth_queue = self._manager.Queue(1)
        self._k_queue = self._manager.Queue(1)
        self._result_queue = self._manager.Queue(1)
        self._happypose_worker = HappyposeWorker(
            pose_estimator,
            inference_params,
            self._device,
            self._worker_free,
            self._stop_worker,
            self._image_queue,
            self._depth_queue,
            self._k_queue,
            self._result_queue,
        )
        self._happypose_worker.start()

        # Create list of camera subscribers
        self._camera_timeout = (
            self.get_parameter(cam_ns + "/timeout").get_parameter_value().double_value
        )
        self._camera_min_num = (
            self.get_parameter(cam_ns + "/min_num").get_parameter_value().integer_value
        )
        camera_names = (
            self.get_parameter(cam_ns + "/cameras")
            .get_parameter_value()
            .string_array_value
        )
        # Each camera registers it's topics and fires synchronisation callback on new image
        self._cameras = {
            name: CameraWrapper(self, name, cam_ns, self._on_image_cb)
            for name in camera_names
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
