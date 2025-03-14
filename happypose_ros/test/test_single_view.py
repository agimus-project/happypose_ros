#!/usr/bin/env python

import pytest
import torch

import launch_ros.actions
import launch_testing.actions
import launch_testing.markers

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from happypose_testing_utils import create_camera_reliable_qos_config
from single_view_base import SingleViewBase


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
            "test_single_view.yaml",
        ]
    )

    # Spawn the happypose_ros node
    ns = "test_single_view"
    happypose_node = launch_ros.actions.Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        namespace=ns,
        parameters=[
            # Dynamically set device and expect raw images
            {"device": device, "cameras.cam_1.compressed": False},
            create_camera_reliable_qos_config(ns, "cam_1", False),
            happypose_params_path,
        ],
    )

    return LaunchDescription(
        [
            happypose_node,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestHappyposeSingleViewNode(SingleViewBase):
    @classmethod
    def setUpClass(cls) -> None:
        # Setup the test case to run with compressed images
        super().setUpClass(namespace="test_single_view", use_compressed=False)
