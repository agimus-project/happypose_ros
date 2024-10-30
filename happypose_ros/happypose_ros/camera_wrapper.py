import math
import numpy as np
import numpy.typing as npt
from typing import Callable, Union, TypeVar

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos_overriding_options import QoSOverridingOptions

from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from cv_bridge import CvBridge
from image_geometry.cameramodels import PinholeCameraModel

from message_filters import ApproximateTimeSynchronizer, Subscriber

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

        self._image = None
        self._camera_info = None
        self._cvb = CvBridge()
        self._cam_model = PinholeCameraModel()

        sync_topics = [
            Subscriber(
                self._node,
                img_msg_type,
                self._camera_name + topic_postfix,
                qos_profile=qos_profile_sensor_data,
                qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
            Subscriber(
                self._node,
                CameraInfo,
                self._camera_name + "/camera_info",
                qos_profile=qos_profile_sensor_data,
                qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
        ]

        # Create time approximate time synchronization
        self._image_approx_time_sync = ApproximateTimeSynchronizer(
            sync_topics,
            queue_size=5,
            slop=params.get_entry(self._camera_name).time_sync_slop,
        )
        # Register callback depending on the configuration
        self._image_approx_time_sync.registerCallback(self._on_image_data_cb)

    def data_recieved_guarded(func: Callable[..., RetType]) -> Callable[..., RetType]:
        """Decorator, checks if data was already received.

        :param func: Function to wrap.
        :type func: Callable[..., RetType]
        :raises RuntimeError: No data was received yet.
        :return: Wrapped function.
        :rtype: Callable[..., RetType]
        """

        def _data_recieved_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._image is None and self._camera_info is None:
                raise RuntimeError(
                    f"No data received yet from the camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _data_recieved_guarded_inner

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

    def _on_image_data_cb(
        self, image: Union[Image, CompressedImage], info: CameraInfo
    ) -> None:
        """Called on every time synchronized image and camera info are received.
        Saves the image and checks if intrinsics are correct. If all checks pass
        calls :func:`_image_sync_hook`.

        :param image: Image received from the camera
        :type image: Union[sensor_msgs.msg.Image, sensor_msgs.msg.CompressedImage]
        :param info: Camera info message
        :type info: sensor_msgs.msg.CameraInfo
        """

        if self._validate_k_matrix(info.k):
            self._camera_info = info
        else:
            self._node.get_logger().warn(
                f"K matrix from topic '{self._info_sub.topic_name()}' is incorrect!",
                throttle_duration_sec=5.0,
            )
            return

        self._image = image
        self._image_sync_hook()

    def ready(self) -> bool:
        """Checks if the camera has all the data needed to use.

        :return: Camera image and intrinsic matrix are available
        :rtype: bool
        """
        return self._image is not None

    @data_recieved_guarded
    def get_last_image_frame_id(self) -> str:
        """Returns frame id associated with the last received image.

        :raises RuntimeError: No images were received yet.
        :return: String with a name of the frame id.
        :rtype: str
        """
        return self._image.header.frame_id

    @data_recieved_guarded
    def get_last_image_stamp(self) -> Time:
        """Returns time stamp associated with last received image.

        :raises RuntimeError: No images were received yet.
        :return: Timestamp of the last received image.
        :rtype: rclpy.Time
        """
        return Time.from_msg(self._image.header.stamp)

    @data_recieved_guarded
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

    @data_recieved_guarded
    def get_last_k_matrix(self) -> npt.NDArray[np.float64]:
        """Returns intrinsic matrix associated with last received camera info message.

        :raises RuntimeError: No camera info messages were received yet.
        :return: 3x3 Numpy array with intrinsic matrix.
        :rtype: numpy.typing.NDArray[numpy.float64]
        """
        self._cam_model.fromCameraInfo(self._camera_info)
        return np.array(self._cam_model.intrinsicMatrix())
