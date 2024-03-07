import time
import functools

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from happypose_ros.happypose_ros.cosypose_loader import CosyPoseLoader
from happypose_ros.happypose_ros.camera_wrapper import CameraWrapper


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
        self._detector = loader.load_detector(self, self._device)
        self._pose_estimator = loader.load_pose_estimator(self, self._device)

        self._pose_estimate_task = None

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
        self._cameras = [
            CameraWrapper(self, name, cam_ns, self._on_image_cb)
            for name in camera_names
        ]

    def _on_image_cb(self) -> None:
        # Skipp if executor is not initialized
        if not self.executor:
            return

        # Skipp if task was initialized and it is already running
        if self._pose_estimate_task and not self._pose_estimate_task.done():
            print(self._pose_estimate_task._callbacks)
            return

        self._pose_estimate_task = self.executor.create_task(
            functools.partial(self._pose_estimate_cb, 5.0)
        )

    def _pose_estimate_cb(self, sleep_time: float) -> None:
        self.get_logger().info(
            f"Blocking pipeline mock begin. Sleeping {sleep_time} seconds..."
        )
        time.sleep(sleep_time)
        self.get_logger().info("Blocking pipeline mock end")


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
    main()
