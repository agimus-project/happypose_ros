#!/usr/bin/env python

import mock
import pytest
from typing import List

import rclpy
from rclpy.exceptions import ParameterException
import rclpy.node
from rclpy.parameter import Parameter

from happypose_ros.happypose_node import HappyPoseNode


@pytest.fixture(autouse=True)
def setup_rclpy():
    """Initialize ROS for every test case."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def happypose_params(request: pytest.FixtureRequest) -> dict:
    """Creates dict of parameters used to initialize
    HappyPoseNode in all test cases

    :param request: Request to obtain test case name.
    :type request: pytest.FixtureRequest
    :return: Class constructor arguments used by all test cases.
    :rtype: dict
    """
    return {
        "namespace": request.node.name,
        "use_global_arguments": False,
        "start_parameter_services": False,
    }


@pytest.fixture()
def minial_overwrites() -> List[Parameter]:
    """Crete set minimal set of ROS parameters needed for the ROS node to start correctly.

    :return: Default, minimal parameters for the ROS node to start.
    :rtype: List[rclpy.Parameter]
    """
    return [
        Parameter("cosypose.dataset_name", Parameter.Type.STRING, "ycbv"),
        Parameter("camera_names", Parameter.Type.STRING_ARRAY, ["cam_1"]),
        Parameter("cameras.cam_1.leading", Parameter.Type.BOOL, True),
        Parameter("cameras.cam_1.publish_tf", Parameter.Type.BOOL, False),
    ]


def test_no_leading(happypose_params: dict) -> None:
    with pytest.raises(ParameterException) as excinfo:
        HappyPoseNode(
            **happypose_params,
            parameter_overrides=[
                Parameter("cosypose.dataset_name", Parameter.Type.STRING, "ycbv"),
                Parameter(
                    "camera_names", Parameter.Type.STRING_ARRAY, ["cam_1", "cam_2"]
                ),
                # At least one camera, has to be leading
            ],
        )
    # Check if param name in exception
    assert ".leading" in str(excinfo.value)


@mock.patch("multiprocessing.context.SpawnContext.Process")
def test_minimal(happypose_params: dict, minial_overwrites: List[Parameter]) -> None:
    # Check if node starts correctly with minimal number of required parameters
    happypose_node = HappyPoseNode(
        **happypose_params,
        parameter_overrides=minial_overwrites,
    )
    happypose_node.destroy_node()


def test_multiple_leading(happypose_params: dict) -> None:
    with pytest.raises(ParameterException) as excinfo:
        HappyPoseNode(
            **happypose_params,
            parameter_overrides=[
                Parameter("cosypose.dataset_name", Parameter.Type.STRING, "ycbv"),
                Parameter(
                    "camera_names", Parameter.Type.STRING_ARRAY, ["cam_1", "cam_2"]
                ),
                # Two cameras can't be leading at the same time
                Parameter("cameras.cam_1.leading", Parameter.Type.BOOL, True),
                Parameter("cameras.cam_2.leading", Parameter.Type.BOOL, True),
            ],
        )
    # Check if both cameras pointed in the exception
    params_to_check = ("cameras.cam_1.leading", "cameras.cam_2.leading")
    assert all(par in str(excinfo.value) for par in params_to_check)


def test_leading_publish_tf(happypose_params: dict) -> None:
    with pytest.raises(ParameterException) as excinfo:
        HappyPoseNode(
            **happypose_params,
            parameter_overrides=[
                Parameter("cosypose.dataset_name", Parameter.Type.STRING, "ycbv"),
                Parameter("camera_names", Parameter.Type.STRING_ARRAY, ["cam_1"]),
                Parameter("cameras.cam_1.leading", Parameter.Type.BOOL, True),
                Parameter("cameras.cam_1.publish_tf", Parameter.Type.BOOL, True),
            ],
        )
    # Check if both parameters are pointed in the exception
    params_to_check = ("cameras.cam_1.leading", "cameras.cam_1.publish_tf")
    assert all(par in str(excinfo.value) for par in params_to_check)


def test_device_unknown(
    happypose_params: dict, minial_overwrites: List[Parameter]
) -> None:
    with pytest.raises(ParameterException) as excinfo:
        HappyPoseNode(
            **happypose_params,
            parameter_overrides=[
                *minial_overwrites,
                # CUDA -1 is an invalid device
                Parameter("device", Parameter.Type.STRING, "cuda:-1"),
            ],
        )
    # Check if both parameters are pointed in the exception
    params_to_check = ("cpu", "device")
    assert all(par in str(excinfo.value) for par in params_to_check)
