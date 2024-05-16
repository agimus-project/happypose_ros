import numpy as np
import numpy.typing as npt
import pinocchio as pin
import time
from typing import Any, List, Optional, Tuple, Union
import unittest
import urllib

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from launch_testing_ros import MessagePump

from cv_bridge import CvBridge

from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from sensor_msgs.msg._image import Metaclass_Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, VisionInfo
from visualization_msgs.msg import MarkerArray, Marker

from rcl_interfaces.msg import Parameter as RCL_Parameter
from rcl_interfaces.srv import GetParameters, SetParametersAtomically


class HappyPoseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(
        cls,
        cameras: List[Tuple[str, Union[Image, CompressedImage]]],
        namespace: str,
    ) -> None:
        rclpy.init()

        cls.tested_node_name = "happypose_node"
        cls.node = HappyPoseTesterNode(
            cameras, cls.tested_node_name, namespace=namespace
        )
        cls.message_pump = MessagePump(cls.node)
        cls.message_pump.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.message_pump.stop()
        cls.node.destroy_node()
        rclpy.shutdown()

    def setUp(self) -> None:
        self.node.assert_find_node(self.tested_node_name, timeout=20.0)


class HappyPoseTesterNode(Node):
    def __init__(
        self,
        cameras: List[Tuple[str, Union[Image, CompressedImage]]],
        tested_node_name: str,
        node_name: str = "happypose_tester_node",
        **kwargs,
    ) -> None:
        super().__init__(node_name, **kwargs)

        self._tested_node_name = tested_node_name

        # Create dict with camera topic publishers
        self._cam_pubs = {
            cam[0]: (
                self.create_publisher(
                    cam[1],
                    # Choose topic name based on the type
                    (
                        f"{cam[0]}/image_color"
                        if isinstance(cam[1], Metaclass_Image)
                        else f"{cam[0]}/image_color/compressed"
                    ),
                    10,
                ),
                self.create_publisher(
                    CameraInfo,
                    (f"{cam[0]}/camera_info"),
                    10,
                ),
            )
            for cam in cameras
        }

        # Initialize topic subscribers
        self._sub_topic = {
            "happypose/markers": [],
            "happypose/detections": [],
            "happypose/vision_info": [],
        }
        self._markers_sub = self.create_subscription(
            MarkerArray, "happypose/markers", self._markers_cb, 5
        )
        self._detections_sub = self.create_subscription(
            Detection2DArray, "happypose/detections", self._detections_cb, 5
        )
        self._vision_info_sub = self.create_subscription(
            VisionInfo, "happypose/vision_info", self._vision_info_cb, 5
        )

        # Initialize service clients
        self._set_param_cli = self.create_client(
            SetParametersAtomically,
            f"{self._tested_node_name}/set_parameters_atomically",
        )
        self._get_param_cli = self.create_client(
            GetParameters, f"{self._tested_node_name}/get_parameters"
        )

        # Initialize transforms listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._cvb = CvBridge()

    def _markers_cb(self, msg: MarkerArray) -> None:
        self._sub_topic["happypose/markers"].append(msg)

    def _detections_cb(self, msg: Detection2DArray) -> None:
        self._sub_topic["happypose/detections"].append(msg)

    def _vision_info_cb(self, msg: VisionInfo) -> None:
        self._sub_topic["happypose/vision_info"].append(msg)

    def get_transform(
        self, target_frame: str, source_frame: str, timeout: float = 5.0
    ) -> Transform:
        return self._tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(), Duration(seconds=timeout)
        ).transform

    def get_params(
        self, param_names: List[str], timeout: float = 5.0
    ) -> List[Parameter]:
        start = time.time()
        ready = False
        while time.time() - start < timeout and not ready:
            ready = self._get_param_cli.service_is_ready()
            time.sleep(0.1)

        assert ready, f"Service {self._get_param_cli.srv_name} is not ready!"

        req = GetParameters.Request(names=param_names)
        future = self._get_param_cli.call_async(req)

        # Subtract already passed time from timeout
        timeout -= time.time() - start
        while time.time() - start < timeout and not future.done():
            time.sleep(0.1)

        assert future.done, (
            "Timeout reached when spinning the service "
            f"{self._get_param_cli.srv_name}!"
        )

        assert (
            future.result() is not None
        ), f"Filed to call the service the service {self._set_param_cli.srv_name}!"

        return [
            Parameter.from_parameter_msg(RCL_Parameter(name=name, value=param))
            for name, param in zip(param_names, future.result().values)
        ]

    def set_params(self, parameters: List[Parameter], timeout: float = 5.0) -> None:
        start = time.time()
        ready = False
        while time.time() - start < timeout and not ready:
            ready = self._set_param_cli.service_is_ready()
            time.sleep(0.1)

        assert ready, f"Service {self._set_param_cli.srv_name} is not ready!"

        req = SetParametersAtomically.Request(
            parameters=[param.to_parameter_msg() for param in parameters]
        )
        future = self._set_param_cli.call_async(req)

        # Subtract already passed time from timeout
        timeout -= time.time() - start
        while time.time() - start < timeout and not future.done():
            time.sleep(0.1)

        assert future.done, (
            "Timeout reached when spinning "
            f"the service {self._set_param_cli.srv_name}!"
        )

        assert (
            future.result() is not None
        ), f"Filed to call the service the service {self._set_param_cli.srv_name}!"

        if not future.result().result.successful:
            "Failed to set parameters!"

    def publish_image(
        self,
        cam: str,
        bgr: npt.NDArray[np.uint8],
        K: npt.NDArray[np.float32],
        stamp: Optional[Time] = None,
    ) -> None:
        if stamp is None:
            stamp = self.get_clock().now()

        if isinstance(self._cam_pubs[cam][0].msg_type, Metaclass_Image):
            img_msg = self._cvb.cv2_to_imgmsg(bgr, encoding="rgb8")
        else:
            # Convert BGR to RGB
            rgb = bgr[:, :, ::-1]
            img_msg = self._cvb.cv2_to_compressed_imgmsg(rgb, dst_format="jpg")

        header = Header(frame_id=cam, stamp=stamp.to_msg())
        img_msg.header = header
        info_msg = CameraInfo(
            header=header,
            height=bgr.shape[0],
            width=bgr.shape[1],
            k=K.reshape(-1),
        )
        # Publish messages
        self._cam_pubs[cam][0].publish(img_msg)
        self._cam_pubs[cam][1].publish(info_msg)

    def clear_msg_buffer(self, topics: Optional[List[str]] = None) -> None:
        if topics is None:
            self._sub_topic = {key: [] for key in self._sub_topic.keys()}
        else:
            for topic in self._sub_topic.keys():
                self._sub_topic[topic] = []

    def get_received_message(self, topic: str) -> Any:
        return self._sub_topic[topic][-1]

    def assert_message_received(self, topic: str, timeout: float = 5.0) -> None:
        start = time.time()
        found = False
        while time.time() - start < timeout and not found:
            found = len(self._sub_topic[topic]) > 0
            time.sleep(0.1)
        assert found, f"No messages received on topic '{topic}'!"

    def assert_find_node(self, node_name: str, timeout: float = 5.0) -> None:
        start = time.time()
        found = False
        while time.time() - start < timeout and not found:
            found = node_name in self.get_node_names()
            time.sleep(0.1)
        assert found, f"'{node_name}' node not launched!"

    def assert_node_is_subscriber(self, topic_name: str, timeout: float = 5.0) -> None:
        start = time.time()
        found = False
        while time.time() - start < timeout and not found:
            found = any(
                self._tested_node_name in n.node_name
                for n in self.get_subscriptions_info_by_topic(topic_name)
            )
            time.sleep(0.1)
        assert found, (
            f"'{self._tested_node_name}' node is not a "
            f"subscriber of a topic '{topic_name}'!"
        )

    def assert_node_is_publisher(self, topic_name: str, timeout: float = 5.0) -> None:
        start = time.time()
        found = False
        while time.time() - start < timeout and not found:
            found = any(
                self._tested_node_name in n.node_name
                for n in self.get_publishers_info_by_topic(topic_name)
            )
            time.sleep(0.1)
        assert found, (
            f"'{self._tested_node_name}' node is not a "
            f"publisher of a topic '{topic_name}'!"
        )


def assert_and_find_detection(
    detections: Detection2DArray, class_id: str
) -> Detection2D:
    for det in detections.detections:
        if det.results[0].hypothesis.class_id == class_id:
            return det

    assert False, f"Class '{class_id}' not found in detections!"


def assert_and_find_marker(
    markers: MarkerArray, class_id: str, dataset_name: str
) -> Marker:
    for marker in markers.markers:
        if class_id.lstrip(f"{dataset_name}-") in marker.mesh_resource:
            return marker
    assert False, f"Class '{class_id}' not found in markers!"


def assert_url_exists(url: str) -> None:
    try:
        urllib.request.urlopen(url)
    except Exception:
        f"URL '{url}' doesn't exist!"


def assert_pose_equal(
    pose_1: Pose,
    pose_2: Pose,
    precision: float = 0.3,
) -> bool:
    poses = [
        pin.XYZQUATToSE3(
            [
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ]
        )
        for p in (pose_1, pose_2)
    ]
    diff = poses[0].inverse() * poses[1]
    assert (
        np.linalg.norm(pin.log6(diff).vector) < precision
    ), "Detected pose is not within specified precision!"


def assert_bbox(
    msg: BoundingBox2D, bbox: List[int], percent_error: float = 30.0
) -> None:
    center_x = bbox[0] + (bbox[2] / 2.0)
    center_y = bbox[1] + (bbox[3] / 2.0)
    size_x = bbox[2]
    size_y = bbox[3]

    pixel_error_x = size_x * percent_error / 100.0
    pixel_error_y = size_y * percent_error / 100.0

    def _almost_equal(x: float, y: float, delta: float) -> bool:
        return abs(x - y) < delta

    assert _almost_equal(
        msg.size_x, size_x, pixel_error_x
    ), "Bbox size in X is incorrect!"

    assert _almost_equal(
        msg.size_y, size_y, pixel_error_y
    ), "Bbox size in Y is incorrect!"

    assert _almost_equal(
        msg.center.position.x, center_x, pixel_error_x
    ), "Bbox center in X is incrrect!"

    assert _almost_equal(
        msg.center.position.y, center_y, pixel_error_y
    ), "Bbox center in Y is incorrect!"

    if abs(msg.center.theta) > 1e-8:
        "Bbox theta is not 0.0!"


def assert_transform_equal(
    transform_1: Transform, transform_2: Transform, precision: float = 0.3
) -> None:
    poses = [
        pin.XYZQUATToSE3(
            [
                t.translation.x,
                t.translation.y,
                t.translation.z,
                t.rotation.x,
                t.rotation.y,
                t.rotation.z,
                t.rotation.w,
            ]
        )
        for t in (transform_1, transform_2)
    ]
    diff = poses[0].inverse() * poses[1]
    assert (
        np.linalg.norm(pin.log6(diff).vector) < precision
    ), "Given transformations are is not within specified precision!"
