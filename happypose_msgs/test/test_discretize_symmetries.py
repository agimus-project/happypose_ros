#!/usr/bin/env python

import numpy as np
import pinocchio as pin
from typing import List

from geometry_msgs.msg import Transform, Vector3, Quaternion

from happypose_msgs_py.symmetries import discretize_symmetries

from happypose_msgs.msg import ObjectSymmetries


def are_transforms_close(t1: Transform, t2: Transform) -> bool:
    """Checks if two ROS transform messages are close.

    :param t1: First transform to compare.
    :type t1: geometry_msgs.msg.Transform
    :param t2: Second transform to compare.
    :type t2: geometry_msgs.msg.Transform
    :return: If those transformations are close.
    :rtype: bool
    """
    T1, T2 = [
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
        for t in (t1, t2)
    ]
    diff = T1.inverse() * T2
    return np.linalg.norm(pin.log6(diff).vector) < 1e-6


def is_transform_in_list(t1: Transform, t_list: List[Transform]) -> bool:
    """Checks if a transform is in the list of transformations.

    :param t1: Transform to check if in the list.
    :type t1: Transform
    :param t_list: List of transforms to check if ``t1`` exists in there.
    :type t_list: List[Transform]
    :return: If the transform in the list.
    :rtype: bool
    """
    return any(are_transforms_close(t1, t2) for t2 in t_list)


def test_empty_message_np() -> None:
    msg = ObjectSymmetries(
        symmetries_discrete=[],
        symmetries_continuous=[],
    )

    res = discretize_symmetries(msg)
    assert isinstance(res, np.ndarray), "Result is not an instance of Numpy array!"
    assert res.shape == (0, 4, 4), "Result shape is incorrect!"


def test_empty_message_ros() -> None:
    msg = ObjectSymmetries(
        symmetries_discrete=[],
        symmetries_continuous=[],
    )

    res = discretize_symmetries(msg, return_ros_msg=True)
    assert isinstance(res, list), "Result is not an instance of a list!"
    assert len(res) == 0, "Results list is not empty!"


def test_only_discrete_ros() -> None:
    msg = ObjectSymmetries(
        symmetries_discrete=[
            Transform(
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            Transform(
                translation=Vector3(x=0.1, y=0.1, z=0.1),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ],
        symmetries_continuous=[],
    )

    res = discretize_symmetries(msg, return_ros_msg=True)

    assert len(res) == len(
        msg.symmetries_discrete
    ), "Results list does not have all discrete symmetries from message received!"

    assert all(
        isinstance(r, Transform) for r in res
    ), "Returned type of elements in the list is not geometry_msgs.msg.Transform!"

    assert all(
        is_transform_in_list(t, msg.symmetries_discrete) for t in res
    ), "Resulted discrete symmetries are not close to the initial ones int the message!"
