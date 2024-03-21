import math
import numpy as np
import numpy.typing as npt
from typing import Callable, Union

from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from cv_bridge import CvBridge


class CameraWrapper:
    def __init__(
        self, node: Node, params, name: str, image_sync_hook: Callable
    ) -> None:
        self._image_sync_hook = image_sync_hook

        self._node = node
        self._camera_name = name

        if params.get_entry(self._camera_name).compressed:
            img_msg_type = CompressedImage
            topic_postfix = "/compressed"
        else:
            img_msg_type = Image
            topic_postfix = "_raw"

        # Define name of the topic
        if params.get_entry(self._camera_name).image_topic:
            image_topic = params.get_entry(self._camera_name).image_topic
        else:
            image_topic = self._camera_name + topic_postfix

        # Define name of the topic
        if params.get_entry(self._camera_name).info_topic:
            self._info_topic = params.get_entry(self._camera_name).info_topic
        else:
            self._info_topic = f"{self._camera_name}/info"

        self._camera_k = None
        # If param with K matrix is correct, assume it is fixed
        param_k_matrix = np.array(params.get_entry(self._camera_name).k_matrix)
        self._fixed_k = self._validate_k_matrix(param_k_matrix)
        if self._fixed_k:
            self._camera_k = param_k_matrix
            self._node.get_logger().info(
                f"Camera '{self._camera_name}' uses fixed K matrix."
                + f" Topic '{self._info_topic}' is not subscribed."
            )
        if not self._fixed_k and np.any(np.nonzero(param_k_matrix)):
            self._node.get_logger().warn(
                f"K matrix for '{self._camera_name}' is incorrect."
                + f" Expecting data on topic '{self._info_topic}'."
            )

        self._image = None
        self._cvb = CvBridge()

        self._image_sub = node.create_subscription(
            img_msg_type, image_topic, self._image_cb, 5
        )

        if not self._fixed_k:
            self._info_sub = node.create_subscription(
                CameraInfo, self._info_topic, self._camera_info_cb, 5
            )

    def _validate_k_matrix(self, k_arr: npt.NDArray[np.float64]) -> bool:
        # Check if parameters expected to be non-zero are in fact non-zero or constant
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
                f"K matrix from topic '{self._info_topic}' is incorrect!"
                + f" Fix it or set 'k_matrix' param for the camera '{self._camera_name}'.",
                throttle_duration_sec=5.0,
            )

    def get_last_iamge_frame_id(self) -> str:
        if not self._image:
            raise ValueError(
                f"No images received yet by camera: '{self._camera_name}'!"
            )
        return self._image.header.frame_id

    def get_last_image_stamp(self) -> Time:
        if not self._image:
            raise ValueError(
                f"No images received yet by camera: '{self._camera_name}'!"
            )
        return Time.from_msg(self._image.header.stamp)

    def get_camera_data(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        if not self._image:
            raise ValueError(
                f"No images received yet by camera: '{self._camera_name}'!"
            )
        if self._camera_k is None:
            raise ValueError(
                f"Camera info was not received yet for camera: '{self._camera_name}'!"
            )

        encoder = (
            self._cvb.imgmsg_to_cv2
            if isinstance(self._image, Image)
            else self._cvb.compressed_imgmsg_to_cv2
        )
        desired_encoding = "passthrough" if self._image.encoding == "rgb8" else "rgb8"
        encoded_image = encoder(self._image, desired_encoding)
        return (self._camera_k, encoded_image)
