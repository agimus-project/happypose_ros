#!/usr/bin/env python

import numpy as np
import PIL.Image
import time
import pytest
import torch

import launch_ros.actions
import launch_testing.actions
import launch_testing.markers

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from happypose_testing_utils import create_camera_reliable_qos_config


from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import Image

from ament_index_python.packages import get_package_share_directory

from launch_testing.io_handler import ActiveIoHandler

from happypose_testing_utils import (
    HappyPoseTestCase,
    assert_and_find_detection,
    assert_pose_equal,
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
            "test_single_view_icp.yaml",
        ]
    )

    # Spawn the happypose_ros node
    ns = "test_single_view_icp"
    happypose_node = launch_ros.actions.Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        namespace=ns,
        parameters=[
            # Dynamically set device and expect raw images
            {"device": device, "cameras.cam_1.compressed": False},
            create_camera_reliable_qos_config(ns, "cam_1", False, False),
            create_camera_reliable_qos_config(
                ns, "cam_1", False, True
            ),  # create qos config for depth cam too
            happypose_params_path,
        ],
    )

    return LaunchDescription(
        [
            happypose_node,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestHappyposeSingleViewNode(HappyPoseTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """

        :raises unittest.SkipTest: Used to prevent the base class from executing as a separate test case.
        """

        super().setUpClass(
            [("cam_1", Image, True)],
            "test_single_view_icp",
        )
        image_path = get_package_share_directory("happypose_ros") + "/test"
        cls.rgb = np.asarray(PIL.Image.open(image_path + "/rgb/000629.png"))
        # https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        # "Multiply the depth image with this factor to get depth in mm"
        # ycbv "depth_scale" is 0.1
        scale = 0.1 / 1000  # image units -> meters
        cls.depth = scale * np.array(
            PIL.Image.open(image_path + "/depth/000629.png"), dtype=np.float32
        )

        cls.K = np.array(
            [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]
        )

        # Based on ground truth, object poses for image 629
        cls.ycbv_02_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0552, -0.0913, 1.0283]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2279, 0.1563, 0.0245, 0.9607]))
            ),
        )
        cls.ycbv_05_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0946, -0.0123, 0.8399]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [-0.4171, 0.7404, -0.4506, -0.273]))
            ),
        )
        cls.ycbv_15_pose = Pose(
            position=Point(**dict(zip("xyz", [-0.1013, 0.0329, 0.9138]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2526, 0.4850, 0.7653, -0.3392]))
            ),
        )

    def test_1_single_view_icp_refine(self, proc_output: ActiveIoHandler) -> None:
        # Clear old messages
        self.node.clear_msg_buffer()

        # Publish images several times to ensure they are received
        start = time.time()
        timeout = 30.0
        ready = False
        # Wait for the first pipeline to be triggered
        while time.time() - start < timeout and not ready:
            self.node.publish_image("cam_1", self.rgb, self.K, self.depth)
            ready = proc_output.waitFor("HappyPose initialized", timeout=0.5)
        assert ready, "Failed to trigger the pipeline!"

        # Publish new to trigger parameter change
        proc_output.assertWaitFor("Parameter change occurred", timeout=0.5)

        self.node.assert_message_received("happypose/detections", timeout=20.0)
        detections = self.node.get_received_message("happypose/detections")

        self.assertEqual(
            detections.header.frame_id, "cam_1", "Incorrect frame_id in the header!"
        )
        self.assertGreaterEqual(
            len(detections.detections), 3, "Incorrect number of detected objects!"
        )
        ycbv_02 = assert_and_find_detection(detections, "ycbv-obj_000002")
        ycbv_05 = assert_and_find_detection(detections, "ycbv-obj_000005")
        ycbv_15 = assert_and_find_detection(detections, "ycbv-obj_000015")

        assert_pose_equal(
            ycbv_02.results[0].pose.pose, self.ycbv_02_pose, precision=0.1
        )
        assert_pose_equal(
            ycbv_05.results[0].pose.pose, self.ycbv_05_pose, precision=0.1
        )
        assert_pose_equal(
            ycbv_15.results[0].pose.pose, self.ycbv_15_pose, precision=0.1
        )
