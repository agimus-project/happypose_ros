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
    """Object wrapping the camera subscriber. Provides unified interface for raw and
    compressed images. Wraps intrinsic matrix subscriber and overwrite option. Contains
    information about the camera's frame id, and last timestamp of the image.
    """

    def __init__(
        self,
        node: Node,
        params: happypose_ros.Params.cameras,
        name: str,
        image_sync_hook: Callable,
    ) -> None:
        """Initializes CameraWrapper object. Checks values of the ROS parameters,
        configures and creates camera image and info subscribers.

        :param node: ROS node to which subscriptions should be attached.
        :type node: rclpy.Node
        :param params: ROS parameters created by generate_parameter_library.
        :type params: happypose_ros.Params.cameras
        :param name: Name of the camera to be registered.
        :type name: str
        :param image_sync_hook: Callback function when a new image arrives.
        :type image_sync_hook: Callable
        :raises rclpy.ParameterException: Intrinsic matrix is incorrect.
        """

        self._image_sync_hook = image_sync_hook

        self._node = node
        self._camera_name = name

        if params.get_entry(self._camera_name).compressed:
            img_msg_type = CompressedImage
            topic_postfix = "/image_raw/compressed"
        else:
            img_msg_type = Image
            topic_postfix = "/image_raw"

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
        """Decorator, checks if image was already received.

        :param func: Function to wrap.
        :type func: Callable[..., RetType]
        :raises RuntimeError: No images were received yet.
        :return: Wrapped function.
        :rtype: Callable[..., RetType]
        """

        def _image_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._image is None:
                raise RuntimeError(
                    f"No images received yet from camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _image_guarded_inner

    def k_matrix_guarded(func: Callable[..., RetType]) -> Callable[..., RetType]:
        """Decorator, checks if an intrinsic matrix was already received.

        :param func: Function to wrap.
        :type func: Callable[..., RetType]
        :raises RuntimeError: No images were received yet.
        :return: Wrapped function.
        :rtype: Callable[..., RetType]
        """

        def _k_matrix_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._camera_k is None:
                raise RuntimeError(
                    f"Camera info was not received yet from camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _k_matrix_guarded_inner

    def _validate_k_matrix(self, k_arr: npt.NDArray[np.float64]) -> bool:
        """Performs basic check of the structure of intrinsic matrix.

        :param k_arr: Intrinsic matrix in a form of a flat Numpy array.
        :type k_arr: numpy.typing.NDArray[numpy.float64]
        :return: Indication if the matrix passed the test.
        :rtype: bool
        """

        # Check if parameters expected to be non-zero are, in fact, non-zero or constant
        keep_vals = [True, False, True, False, True, True, False, False, True]
        return np.all(k_arr[keep_vals] > 0.0) and math.isclose(k_arr[-1], 1.0)

    def _image_cb(self, image: Union[Image, CompressedImage]) -> None:
        """Called on every received image. Saves the image. If intrinsic matrix
        already received calls :func:`_image_sync_hook`

        :param image: Image received from the camera
        :type image: Union[sensor_msgs.msg.Image, sensor_msgs.msg.CompressedImage]
        """

        self._image = image
        # Fire the callback only if camera info is available
        if self._camera_k is not None:
            self._image_sync_hook()

    def _camera_info_cb(self, info: CameraInfo) -> None:
        """Receives camera info messages and extracts intrinsic matrices from them.

        :param info: Camera info message
        :type info: sensor_msgs.msg.CameraInfo
        """

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
        """Checks if the camera has all the data needed to use.

        :return: Camera image and intrinsic matrix are available
        :rtype: bool
        """
        return self._image is not None and self._camera_k is not None

    @image_guarded
    def get_last_image_frame_id(self) -> str:
        """Returns frame id associated with the last received image.

        :raises RuntimeError: No images were received yet.
        :return: String with a name of the frame id.
        :rtype: str
        """
        return self._image.header.frame_id

    @image_guarded
    def get_last_image_stamp(self) -> Time:
        """Returns time stamp associated with last received image.

        :raises RuntimeError: No images were received yet.
        :return: Timestamp of the last received image.
        :rtype: rclpy.Time
        """
        return Time.from_msg(self._image.header.stamp)

    @image_guarded
    def get_last_rgb_image(self) -> npt.NDArray[np.uint8]:
        """Returns last received color image.

        :raises RuntimeError: No images were received yet.
        :return: Image converted to OpenCV format in 'rgb8' encoding.
        :rtype: numpy.typing.NDArray[numpy.uint8]
        """
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
        """Returns intrinsic matrix associated with last received camera info message.

        :raises RuntimeError: No camera info messages were received yet.
        :return: 3x3 Numpy array with intrinsic matrix.
        :rtype: numpy.typing.NDArray[numpy.float64]
        """
        return self._camera_k.reshape((3, 3))
