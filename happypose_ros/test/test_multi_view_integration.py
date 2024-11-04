#!/usr/bin/env python

import numpy as np
import PIL.Image
import pytest
import time
import torch
from typing import List

from rclpy.clock import ClockType
from rclpy.constants import S_TO_NS
from rclpy.parameter import Parameter
from rclpy.time import Time


from geometry_msgs.msg import Point, Pose, Transform, Vector3, Quaternion
from sensor_msgs.msg import Image, CompressedImage

from ament_index_python.packages import get_package_share_directory

import launch_ros.actions
import launch_testing.actions
import launch_testing.markers

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from launch_testing.io_handler import ActiveIoHandler

from happypose_testing_utils import (
    HappyPoseTestCase,
    assert_and_find_detection,
    assert_pose_equal,
    assert_transform_equal,
    create_camera_reliable_qos_config,
)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # Assume testing machine has only one GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Find parameters file
    happypose_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_ros"),
            "test",
            "test_multi_view.yaml",
        ]
    )

    # Spawn the happypose_ros node
    ns = "test_multi_view"
    happypose_node = launch_ros.actions.Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        namespace=ns,
        # Dynamically set device
        parameters=[
            {"device": device},
            *[create_camera_reliable_qos_config(ns, f"cam_{i}") for i in range(3)],
            happypose_params_path,
        ],
    )

    return LaunchDescription(
        [
            happypose_node,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestHappyposeTesterMultiViewNode(HappyPoseTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Test both multiview and compressed images
        super().setUpClass(
            [("cam_1", Image), ("cam_2", CompressedImage), ("cam_3", Image)],
            "test_multi_view",
        )

        cls.K = np.array(
            [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]
        )

        image_path = get_package_share_directory("happypose_ros") + "/test"
        cls.cam_1_image = np.asarray(PIL.Image.open(image_path + "/000629.png"))
        cls.cam_2_image = np.asarray(PIL.Image.open(image_path + "/001130.png"))
        cls.cam_3_image = np.asarray(PIL.Image.open(image_path + "/001874.png"))

    def test_01_node_startup(self, proc_output: ActiveIoHandler) -> None:
        # Check if the node outputs correct initialization
        proc_output.assertWaitFor("Node initialized", timeout=20.0)

    def test_02_check_topics(self) -> None:
        # Check if node subscribes to correct topics
        self.node.assert_node_is_subscriber("cam_1/image_raw", timeout=3.0)
        self.node.assert_node_is_subscriber("cam_1/camera_info", timeout=3.0)
        self.node.assert_node_is_subscriber("cam_2/image_raw/compressed", timeout=3.0)
        self.node.assert_node_is_subscriber("cam_2/camera_info", timeout=3.0)
        self.node.assert_node_is_subscriber("cam_3/image_raw", timeout=3.0)
        self.node.assert_node_is_subscriber("cam_3/camera_info", timeout=3.0)
        self.node.assert_node_is_publisher("happypose/detections", timeout=3.0)
        self.node.assert_node_is_publisher("happypose/vision_info", timeout=3.0)
        self.node.assert_node_is_publisher("/tf", timeout=3.0)

    def test_03_check_not_publish_markers(self) -> None:
        with self.assertRaises(AssertionError) as excinfo:
            self.node.assert_node_is_publisher("happypose/markers", timeout=3.0)
        self.assertTrue(
            "node is not a publisher of" in str(excinfo.exception),
            msg="One image after timeout triggered the pipeline!",
        )

    def test_04_trigger_pipeline(self, proc_output: ActiveIoHandler) -> None:
        # Clear buffer before expecting any messages
        self.node.clear_msg_buffer()

        # Publish images several times to ensure they are received
        start = time.time()
        timeout = 30.0
        ready = False
        # Wait for the first pipeline to be triggered
        while time.time() - start < timeout and not ready:
            self.node.publish_image("cam_1", self.cam_1_image, self.K)
            self.node.publish_image("cam_2", self.cam_2_image, self.K)
            self.node.publish_image("cam_3", self.cam_3_image, self.K)
            ready = proc_output.waitFor("HappyPose initialized", timeout=0.5)
        if not ready:
            self.fail("Failed to trigger the pipeline!")

    def test_05_receive_messages(self) -> None:
        self.node.assert_message_received("happypose/detections", timeout=180.0)
        self.node.assert_message_received("happypose/vision_info", timeout=8.0)

    def test_06_check_vision_info(self) -> None:
        vision_info = self.node.get_received_message("happypose/vision_info")
        self.assertEqual(vision_info.method, "cosypose")
        self.assertTrue("ycbv" in vision_info.database_location)

    def test_07_check_detection(self) -> None:
        detections = self.node.get_received_message("happypose/detections")
        # Ensure none of the detections have bounding box
        for detection in detections.detections:
            bbox = detection.bbox
            self.assertAlmostEqual(
                bbox.size_x, 0.0, places=6, msg="Bounding box size in X is not 0"
            )
            self.assertAlmostEqual(
                bbox.size_x, 0.0, places=6, msg="Bounding box size in Y is not 0"
            )
            self.assertAlmostEqual(
                bbox.center.position.x,
                0.0,
                places=6,
                msg="Bounding box center in X is not 0",
            )
            self.assertAlmostEqual(
                bbox.center.position.y,
                0.0,
                places=6,
                msg="Bounding box center in Y is not 0",
            )
            self.assertAlmostEqual(
                bbox.size_x, 0.0, places=6, msg="Bounding box theta is not 0"
            )

        # At least 3 objects are expected to be detected
        self.assertGreaterEqual(
            len(detections.detections), 3, "Incorrect number of detections!"
        )

        # 3 following objects are known to be decent detections from given images
        ycbv_02 = assert_and_find_detection(detections, "ycbv-obj_000002")
        ycbv_05 = assert_and_find_detection(detections, "ycbv-obj_000005")
        ycbv_15 = assert_and_find_detection(detections, "ycbv-obj_000015")

        # Based on ground truth, object poses for image 629
        ycbv_02_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0552, -0.0913, 1.0283]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2279, 0.1563, 0.0245, 0.9607]))
            ),
        )
        ycbv_05_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0946, -0.0123, 0.8399]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [-0.4171, 0.7404, -0.4506, -0.273]))
            ),
        )
        ycbv_15_pose = Pose(
            position=Point(**dict(zip("xyz", [-0.1013, 0.0329, 0.9138]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2526, 0.4850, 0.7653, -0.3392]))
            ),
        )

        assert_pose_equal(ycbv_02.results[0].pose.pose, ycbv_02_pose)
        assert_pose_equal(ycbv_05.results[0].pose.pose, ycbv_05_pose)
        assert_pose_equal(ycbv_15.results[0].pose.pose, ycbv_15_pose)

    def test_08_check_not_published_transforms(self) -> None:
        self.assertFalse(
            self.node.can_transform("cam_1", "cam_2"),
            msg="`cam_2` frame_id was was published even thought it shouldn't!",
        )

    def test_09_check_transforms_correct(self) -> None:
        # Based on transformed ground truth
        # Image 1874 camera pose transformed into image 629 camera pose reference frame
        expected_translation = Transform(
            translation=Vector3(**dict(zip("xyz", [-0.4790, -0.0166, 0.3517]))),
            rotation=Quaternion(**dict(zip("xyzw", [0.0536, 0.3365, 0.1493, 0.9281]))),
        )
        self.assertTrue(
            self.node.can_transform("cam_1", "cam_3"),
            msg="`cam_3` frame_id was not published!",
        )
        cam_3_trans = self.node.get_transform("cam_1", "cam_3")
        assert_transform_equal(cam_3_trans, expected_translation)

    def push_data(self, stamp: Time) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()
        # Publish three images and expect to pass
        self.node.publish_image("cam_1", self.cam_1_image, self.K, stamp)
        self.node.publish_image("cam_2", self.cam_2_image, self.K, stamp)
        self.node.publish_image("cam_3", self.cam_3_image, self.K, stamp)
        self.node.assert_message_received("happypose/detections", timeout=60.0)

    def test_10_dynamic_params_change_frame_id(self) -> None:
        # Set cam_3 frame_id
        self.node.set_params(
            [
                Parameter(
                    "cameras.cam_3.estimated_tf_frame_id",
                    Parameter.Type.STRING,
                    "custom_cam_3_frame_id",
                ),
            ],
            10.0,
        )

        # Wait more than the timeout
        time.sleep(2.0)
        # Get fresh timestamp
        stamp = self.node.get_clock().now()
        self.push_data(stamp)

        self.assertFalse(
            self.node.can_transform("cam_1", "cam_3", stamp),
            msg="`cam_3` frame_id was was published even thought it shouldn't!",
        )
        self.assertTrue(
            self.node.can_transform("cam_1", "custom_cam_3_frame_id", stamp),
            msg="`custom_cam_3_frame_id` frame_id was not published!",
        )

        # Based on transformed ground truth
        # Image 1874 camera pose transformed into image 629 camera pose reference frame
        expected_translation = Transform(
            translation=Vector3(**dict(zip("xyz", [-0.4790, -0.0166, 0.3517]))),
            rotation=Quaternion(**dict(zip("xyzw", [0.0536, 0.3365, 0.1493, 0.9281]))),
        )
        cam_3_trans = self.node.get_transform("cam_1", "custom_cam_3_frame_id")
        assert_transform_equal(cam_3_trans, expected_translation)

    def test_11_dynamic_params_change_frame_id_to_empty(self) -> None:
        # Use default frame_if for cam_3
        self.node.set_params(
            [
                Parameter(
                    "cameras.cam_3.estimated_tf_frame_id",
                    Parameter.Type.STRING,
                    "",
                ),
            ],
            10.0,
        )

        # Wait more than the timeout
        time.sleep(2.0)
        # Get fresh timestamp
        stamp = self.node.get_clock().now()
        self.push_data(stamp)

        self.assertFalse(
            self.node.can_transform("cam_1", "custom_cam_3_frame_id", stamp),
            msg="`custom_cam_3_frame_id` frame_id was was published even thought it shouldn't!",
        )
        self.assertTrue(
            self.node.can_transform("cam_1", "cam_3", stamp),
            msg="`cam_3` frame_id was not published!",
        )

    def set_timeout(self, timeout: float) -> None:
        self.node.set_params(
            [
                Parameter(
                    "cameras.timeout",
                    Parameter.Type.DOUBLE,
                    timeout,
                )
            ],
            timeout=10.0,
        )

    def test_12_dynamic_params_camera_no_timeout(self) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()

        # Disable timeout. Single image should trigger now
        self.set_timeout(0.0)
        self.node.publish_image("cam_1", self.cam_1_image, self.K)
        self.node.assert_message_received("happypose/detections", timeout=180.0)

    def expect_no_detection(self) -> None:
        with self.assertRaises(AssertionError) as excinfo:
            self.node.assert_message_received("happypose/detections", timeout=60.0)
        self.assertTrue(
            "No messages received" in str(excinfo.exception),
            msg="One image after timeout triggered the pipeline!",
        )

    def test_13_dynamic_params_camera_timeout_one_camera(self) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()
        # Wait more than the timeout
        time.sleep(1.0)
        # Enable timeout
        self.set_timeout(0.5)
        # Publish one image
        self.node.publish_image("cam_1", self.cam_1_image, self.K)
        self.expect_no_detection()

    def test_14_dynamic_params_camera_timeout_two_cameras(self) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()
        # Wait more than the timeout
        time.sleep(1.0)
        # Publish two images
        self.node.publish_image("cam_1", self.cam_1_image, self.K)
        self.node.publish_image("cam_2", self.cam_2_image, self.K)
        self.expect_no_detection()

    def test_15_dynamic_params_camera_timeout_three_cameras_ok(self) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()
        # Wait more than the timeout
        time.sleep(1.0)
        # Publish three images and expect to pass
        self.node.publish_image("cam_1", self.cam_1_image, self.K)
        self.node.publish_image("cam_2", self.cam_2_image, self.K)
        self.node.publish_image("cam_3", self.cam_3_image, self.K)
        self.node.assert_message_received("happypose/detections", timeout=60.0)

    def test_16_dynamic_params_camera_timeout_three_cameras_short(self) -> None:
        # Set timeout to a small value
        timeout = 0.05
        self.set_timeout(timeout)
        # Wait more than the timeout
        time.sleep(1.0)
        # Clear old messages
        self.node.clear_msg_buffer()
        # Publish images in loop, do not expect any change
        for _ in range(8):
            for _ in range(5):
                self.node.publish_image("cam_1", self.cam_1_image, self.K)
                time.sleep(timeout * 3.0)
                self.node.publish_image("cam_2", self.cam_2_image, self.K)
                time.sleep(timeout * 3.0)
                self.node.publish_image("cam_3", self.cam_3_image, self.K)
            self.expect_no_detection()

    def test_17_dynamic_params_camera_timeout_three_cameras_short_ok(self) -> None:
        # Set timeout to a small value
        self.set_timeout(0.05)
        # Wait more than the timeout
        time.sleep(1.0)
        # Clear old messages
        self.node.clear_msg_buffer()
        # Publish many images expecting them to be within timeout
        self.node.publish_image("cam_1", self.cam_1_image, self.K)
        self.node.publish_image("cam_2", self.cam_2_image, self.K)
        self.node.publish_image("cam_3", self.cam_3_image, self.K)
        self.node.assert_message_received("happypose/detections", timeout=60.0)

    def setup_timestamp_test(
        self, offsets: List[float], expected: float, strategy: str
    ) -> None:
        now = self.node.get_clock().now()
        cam_1_stamp = now - Time(seconds=offsets[0], clock_type=ClockType.ROS_TIME)
        cam_2_stamp = now - Time(seconds=offsets[1], clock_type=ClockType.ROS_TIME)
        cam_3_stamp = now - Time(seconds=offsets[2], clock_type=ClockType.ROS_TIME)
        cam_1_stamp = Time(nanoseconds=cam_1_stamp.nanoseconds)
        cam_2_stamp = Time(nanoseconds=cam_2_stamp.nanoseconds)
        cam_3_stamp = Time(nanoseconds=cam_3_stamp.nanoseconds)

        # Clear old messages
        self.node.clear_msg_buffer()
        # Prevent the pipeline from triggering
        self.set_timeout(0.00001)
        # Change the strategy
        self.node.set_params(
            [
                Parameter(
                    "time_stamp_strategy",
                    Parameter.Type.STRING,
                    strategy,
                ),
            ],
            10.0,
        )

        self.node.publish_image("cam_1", self.cam_1_image, self.K, cam_1_stamp)
        self.node.publish_image("cam_2", self.cam_2_image, self.K, cam_2_stamp)
        # Accept any timestamp from the message
        self.set_timeout(0.0)
        self.node.publish_image("cam_3", self.cam_3_image, self.K, cam_3_stamp)
        # Await the results
        self.node.assert_message_received("happypose/detections", timeout=60.0)

        detections = self.node.get_received_message("happypose/detections")
        stamp_sec = (Time.from_msg(detections.header.stamp) - now).nanoseconds / S_TO_NS
        self.assertAlmostEqual(
            stamp_sec,
            expected,
            places=5,
            msg=f"Timestamp was not chosen to be '{strategy}'",
        )

    def test_18_dynamic_params_timestamp_strategy(self) -> None:
        offsets = [0.0, 0.01, 0.02]
        self.setup_timestamp_test(offsets, 0.0, "newest")

    def test_19_dynamic_params_timestamp_strategy(self) -> None:
        offsets = [0.0, 0.01, 0.05]
        # Offestes are subtracted from time current time so
        # the result has to be expected in the past
        self.setup_timestamp_test(offsets, -0.05, "oldest")

    def test_20_dynamic_params_timestamp_strategy(self) -> None:
        offsets = [0.0, 0.01, 0.05]
        self.setup_timestamp_test(offsets, -0.02, "average")
