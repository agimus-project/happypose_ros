import math
import numpy as np
import numpy.typing as npt
from typing import Callable, Union, TypeVar

from rclpy.exceptions import ParameterException
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from cv_bridge import CvBridge

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros

RetType = TypeVar("RetType")


class CameraWrapper:
    def __init__(
        self,
        node: Node,
        params: happypose_ros.Params.cameras,
        name: str,
        image_sync_hook: Callable,
    ) -> None:
        self._image_sync_hook = image_sync_hook

        self._node = node
        self._camera_name = name

        if params.get_entry(self._camera_name).compressed:
            img_msg_type = CompressedImage
            topic_postfix = "/image_color/compressed"
        else:
            img_msg_type = Image
            topic_postfix = "/image_color"

        self._camera_k = None
        # If param with K matrix is correct, assume it is fixed
        param_k_matrix = np.array(params.get_entry(self._camera_name).k_matrix)
        self._fixed_k = self._validate_k_matrix(param_k_matrix)
        if self._fixed_k:
            self._camera_k = param_k_matrix
            self._node.get_logger().warn(
                f"Camera '{self._camera_name}' uses fixed K matrix."
                f" Camera info topic is not subscribed."
            )
        # If non zero, but incorrect
        elif np.any(np.nonzero(param_k_matrix)):
            e = ParameterException(
                f"K matrix for '{self._camera_name}' is incorrect",
                (f"cameras.{self._camera_name}.k_matrix"),
            )
            self._node.get_logger().error(str(e))
            raise e

        self._image = None
        self._cvb = CvBridge()

        image_topic = self._camera_name + topic_postfix
        self._image_sub = node.create_subscription(
            img_msg_type, image_topic, self._image_cb, 5
        )

        if not self._fixed_k:
            info_topic = self._camera_name + "/camera_info"
            self._info_sub = node.create_subscription(
                CameraInfo, info_topic, self._camera_info_cb, 5
            )

    def image_guarded(func: Callable[..., RetType]) -> Callable[..., RetType]:
        def _image_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._image is None:
                raise RuntimeError(
                    f"No images received yet from camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _image_guarded_inner

    def k_matrix_guarded(func: Callable[..., RetType]) -> Callable[..., RetType]:
        def _k_matrix_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._camera_k is None:
                raise RuntimeError(
                    f"Camera info was not received yet from camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _k_matrix_guarded_inner

    def _validate_k_matrix(self, k_arr: npt.NDArray[np.float64]) -> bool:
        # Check if parameters expected to be non-zero are, in fact, non-zero or constant
        keep_vals = [True, False, True, False, True, True, False, False, True]
        return np.all(k_arr[keep_vals] > 0.0) and math.isclose(k_arr[-1], 1.0)

    def _image_cb(self, image: Union[Image, CompressedImage]) -> None:
        self._image = image
        # Fire the callback only if camera info is available
        if self._camera_k is not None:
            self._image_sync_hook()

    def _camera_info_cb(self, info: CameraInfo) -> None:
        if self._validate_k_matrix(info.k):
            self._camera_k = info.k
        else:
            self._node.get_logger().warn(
                f"K matrix from topic '{self._info_sub.topic_name()}' is incorrect!"
                f" Fix it or set 'cameras.{self._camera_name}.k_matrix'"
                f" param for the camera '{self._camera_name}'.",
                throttle_duration_sec=5.0,
            )

    def ready(self) -> bool:
        return self._image is not None and self._camera_k is not None

    @image_guarded
    def get_last_image_frame_id(self) -> str:
        return self._image.header.frame_id

    @image_guarded
    def get_last_image_stamp(self) -> Time:
        return Time.from_msg(self._image.header.stamp)

    @image_guarded
    def get_last_rgb_image(self) -> npt.NDArray[np.uint8]:
        encoder = (
            self._cvb.imgmsg_to_cv2
            if isinstance(self._image, Image)
            else self._cvb.compressed_imgmsg_to_cv2
        )
        desired_encoding = (
            "passthrough"
            # Compressed image has no attribute "encoding"
            if hasattr(self._image, "encoding") and self._image.encoding == "rgb8"
            else "rgb8"
        )
        return encoder(self._image, desired_encoding)

    @k_matrix_guarded
    def get_last_k_matrix(self) -> npt.NDArray[np.float64]:
        return self._camera_k.reshape((3, 3))
