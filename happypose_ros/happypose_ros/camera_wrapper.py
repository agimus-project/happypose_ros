from typing import Callable, Union
import numpy as np
import numpy.typing as npt

from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from cv_bridge import CvBridge


class CameraWrapper:
    def __init__(
        self, node: Node, params: dict, name: str, image_sync_hook: Callable
    ) -> None:
        self._image_sync_hook = image_sync_hook

        if not params[name].compressed:
            img_msg_type = CompressedImage
            topic_name = "image_compressed"
        else:
            img_msg_type = Image
            topic_name = "image_raw"

        self._camera_info = None
        self._image = None
        self._cvb = CvBridge()

        self._image_sub = node.create_subscription(
            img_msg_type, f"/{name}/{topic_name}", self._camera_cb, 5
        )

        self._info_sub = node.create_subscription(
            img_msg_type, f"/{self._camera_name}/info", self._camera_info_cb, 5
        )

    def _image_cb(self, image: Union[Image, CompressedImage]) -> None:
        self._image = image
        # Fire the callback only if camera info is available
        if self._camera_info:
            self._image_sync_hook()

    def _camera_info_cb(self, info: CameraInfo) -> None:
        self._camera_info = info

    def get_last_image_stamp(self) -> Time:
        if not self._image:
            msg = f"No images received yet by camera: '{self._camera_name}'!"
            raise ValueError(msg)
        return self._image.header.stamp

    def get_camera_data(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        if not self._image:
            msg = f"No images received yet by camera: '{self._camera_name}'!"
            raise ValueError(msg)
        if not self._camera_info:
            msg = f"Camera info was not received yet for camera: '{self._camera_name}'!"
            raise ValueError(msg)

        encoder = (
            self._cvb.imgmsg_to_cv2
            if isinstance(self._image, Image)
            else self._cvb.compressed_imgmsg_to_cv2
        )
        desired_encoding = "passthrough" if self._image.encoding == "rgb8" else "rgb8"
        encoded_image = encoder(self._image, desired_encoding)
        return (self._camera_info.k, encoded_image)
