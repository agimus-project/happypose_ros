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
        use_depth: bool = False,
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
        :param use_depth: Whether to subscribe to depth image topic or not, defaults to True.
        :type use_depth: bool, optional
        :raises rclpy.ParameterException: Intrinsic matrix is incorrect.
        """

        self._image_sync_hook = image_sync_hook

        self._node = node
        self._camera_name = name

        camera_params = params.get_entry(self._camera_name)
        if camera_params.compressed:
            img_msg_type = CompressedImage
            topic_postfix = "/image_raw/compressed"
        else:
            img_msg_type = Image
            topic_postfix = "/image_raw"

        self._color_image: Union[Image, CompressedImage] = None
        self._color_camera_info: CameraInfo = None
        self._depth_image: Union[Image, CompressedImage] = None
        self._depth_camera_info: Union[None, Image, CompressedImage] = None
        self._cvb = CvBridge()
        self._estimated_tf_frame_id = camera_params.estimated_tf_frame_id
        self._cam_model = PinholeCameraModel()

        sync_topics = [
            Subscriber(
                self._node,
                img_msg_type,
                self._camera_name + "/color" + topic_postfix,
                qos_profile=qos_profile_sensor_data,
                qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
            Subscriber(
                self._node,
                CameraInfo,
                self._camera_name + "/color" + "/camera_info",
                qos_profile=qos_profile_sensor_data,
                qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
        ]

        if use_depth:
            sync_topics + [
                Subscriber(
                    self._node,
                    img_msg_type,
                    self._camera_name + "/depth" + topic_postfix,
                    qos_profile=qos_profile_sensor_data,
                    qos_overriding_options=QoSOverridingOptions.with_default_policies(),
                ),
                Subscriber(
                    self._node,
                    CameraInfo,
                    self._camera_name + "/depth/camera_info",
                    qos_profile=qos_profile_sensor_data,
                    qos_overriding_options=QoSOverridingOptions.with_default_policies(),
                ),
            ]

        # Create time approximate time synchronization
        self._color_image_approx_time_sync = ApproximateTimeSynchronizer(
            sync_topics,
            queue_size=5,
            slop=camera_params.time_sync_slop,
        )
        # Register callback depending on the configuration
        self._color_image_approx_time_sync.registerCallback(
            self._on_image_data_cb if use_depth else self._on_image_data_cb
        )

    def update_params(self, params: happypose_ros.Params.cameras) -> None:
        """Updates internal parameters of given camera

        :param params: ROS parameters created by generate_parameter_library.
        :type params: happypose_ros.Params.cameras
        """
        camera_params = params.get_entry(self._camera_name)
        self._estimated_tf_frame_id = camera_params.estimated_tf_frame_id

    def data_received_guarded(func: Callable[..., RetType]) -> Callable[..., RetType]:
        """Decorator, checks if data was already received.

        :param func: Function to wrap.
        :type func: Callable[..., RetType]
        :raises RuntimeError: No data was received yet.
        :return: Wrapped function.
        :rtype: Callable[..., RetType]
        """

        def _data_received_guarded_inner(self, *args, **kwargs) -> RetType:
            if self._color_image is None and self._color_camera_info is None:
                raise RuntimeError(
                    f"No data received yet from the camera '{self._camera_name}'!"
                )
            return func(self, *args, **kwargs)

        return _data_received_guarded_inner

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
        self._on_image_with_depth_data_cb(image, info, None, None)

    def _on_image_with_depth_data_cb(
        self,
        color_image: Union[Image, CompressedImage],
        color_camera_info: CameraInfo,
        depth_image: Union[None, Image, CompressedImage] = None,
        depth_camera_info: Union[None, CameraInfo] = None,
    ) -> None:
        """Called on every time synchronized image and camera info are received.
        Saves the image and checks if intrinsics are correct. If all checks pass
        calls :func:`_image_sync_hook`.

        :param color_image: Image received from the color camera sensor
        :type color_image: Union[sensor_msgs.msg.Image, sensor_msgs.msg.CompressedImage]
        :param color_info: Camera info message for the color camera sensor.
        :type color_info: sensor_msgs.msg.CameraInfo
        :param color_image: Image received from the depth camera sensor. None if not used.
        :type color_image: Union[None, sensor_msgs.msg.Image, sensor_msgs.msg.CompressedImage]
        :param color_info: Camera info message for the depth camera sensor. None if not used.
        :type color_info: Union[None, sensor_msgs.msg.CameraInfo]
        """

        connections = self._color_image_approx_time_sync.input_connections
        frame_ids = {color_image.header.frame_id, color_camera_info.header.frame_id}
        if len(frame_ids) > 1:
            self._node.get_logger().warn(
                "Mismatch in `frame_id` between topics "
                + f"'{connections[0].getTopic()}' and '{connections[1].getTopic()}'!",
                throttle_duration_sec=5.0,
            )
            return

        if self._validate_k_matrix(color_camera_info.k):
            self._color_camera_info = color_camera_info
        else:
            topic = self._color_image_approx_time_sync.input_connections[1].getTopic()
            self._node.get_logger().warn(
                f"K matrix from topic '{topic}' is incorrect!",
                throttle_duration_sec=5.0,
            )
            return

        if depth_camera_info:
            if not np.allclose(color_camera_info.k, depth_camera_info.k):
                self._node.get_logger().warn(
                    f"Topics '{connections[1].getTopic()}' and "
                    + f"'{connections[3].getTopic()}' contain different intrinsics matrices! "
                    + "Both color and depth images have to have the same intrinsics for ICP to work!",
                    throttle_duration_sec=5.0,
                )
                return

            depth_frame_ids = {
                depth_image.header.frame_id,
                depth_camera_info.header.frame_id,
            }
            if len(depth_frame_ids) > 1:
                self._node.get_logger().warn(
                    "Mismatch in `frame_id` between topics "
                    + f"'{connections[2].getTopic()}' and '{connections[3].getTopic()}'!",
                    throttle_duration_sec=5.0,
                )
                return

            if len(depth_frame_ids) > 1:
                self._node.get_logger().warn(
                    "Mismatch in `frame_id` between topics "
                    + f"'{connections[2].getTopic()}' and '{connections[3].getTopic()}'!",
                    throttle_duration_sec=5.0,
                )
                return

            if len(frame_ids | depth_frame_ids) > 1:
                self._node.get_logger().warn(
                    f"Topics '{connections[0].getTopic()}' and "
                    + f"'{connections[2].getTopic()}' contain images with different `frame_id`! "
                    + "Depth image should be projected to mach frame of the color image for ICP to work!",
                    throttle_duration_sec=5.0,
                )
                return

        self._color_image = color_image
        self._depth_image = depth_image

        self._image_sync_hook()

    def ready(self) -> bool:
        """Checks if the camera has all the data needed to use.

        :return: Camera image and intrinsic matrix are available
        :rtype: bool
        """
        return self._color_image is not None

    @data_received_guarded
    def get_last_image_frame_id(self) -> str:
        """Returns frame id associated with the camera.

        :raises RuntimeError: No images were received yet.
        :return: String with a name of the frame id.
        :rtype: str
        """
        return (
            self._color_image.header.frame_id
            if self._estimated_tf_frame_id == ""
            else self._estimated_tf_frame_id
        )

    @data_received_guarded
    def get_last_image_stamp(self) -> Time:
        """Returns time stamp associated with last received image.

        :raises RuntimeError: No images were received yet.
        :return: Timestamp of the last received image.
        :rtype: rclpy.Time
        """
        return Time.from_msg(self._color_image.header.stamp)

    @data_received_guarded
    def get_last_rgb_image(self) -> npt.NDArray[np.uint8]:
        """Returns last received color image.

        :raises RuntimeError: No images were received yet.
        :return: Image converted to OpenCV format in 'rgb8' encoding.
        :rtype: numpy.typing.NDArray[numpy.uint8]
        """
        encoder = (
            self._cvb.imgmsg_to_cv2
            if isinstance(self._color_image, Image)
            else self._cvb.compressed_imgmsg_to_cv2
        )
        desired_encoding = (
            "passthrough"
            # Compressed image has no attribute "encoding"
            if hasattr(self._color_image, "encoding")
            and self._color_image.encoding == "rgb8"
            else "rgb8"
        )
        return encoder(self._color_image, desired_encoding)

    @data_received_guarded
    def get_last_k_matrix(self) -> npt.NDArray[np.float64]:
        """Returns intrinsic matrix associated with last received color camera info message.
        If depth is used, both color and depth intrinsics matrices have to be equal.

        :raises RuntimeError: No camera info messages were received yet.
        :return: 3x3 Numpy array with intrinsic matrix.
        :rtype: numpy.typing.NDArray[numpy.float64]
        """
        self._cam_model.fromCameraInfo(self._color_camera_info)
        return np.array(self._cam_model.intrinsicMatrix())

    @data_received_guarded
    def get_last_depth_image(self) -> Union[None, npt.NDArray[np.uint16]]:
        """Returns last received depth image.

        :raises RuntimeError: No images were received yet.
        :return: Image converted to OpenCV format in '16UC1' encoding.
            If depth is not used None is returned.
        :rtype: Union[None, numpy.typing.NDArray[numpy.uint8]]
        """
        if self._depth_image is None:
            return None

        encoder = (
            self._cvb.imgmsg_to_cv2
            if isinstance(self._depth_image, Image)
            else self._cvb.compressed_imgmsg_to_cv2
        )
        desired_encoding = (
            "passthrough"
            # Compressed image has no attribute "encoding"
            if hasattr(self._depth_image, "encoding")
            and self._depth_image.encoding == "16UC1"
            else "16UC1"
        )
        depth_image = encoder(self._depth_image, desired_encoding)
        return (
            depth_image.astype(np.uint16)
            if desired_encoding == "passthrough"
            else depth_image
        )
