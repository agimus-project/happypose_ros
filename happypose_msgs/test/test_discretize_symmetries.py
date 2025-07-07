#!/usr/bin/env python

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from typing import List

from geometry_msgs.msg import Transform, Vector3, Quaternion

from happypose_msgs_py.symmetries import discretize_symmetries

from happypose_msgs.msg import ContinuousSymmetry, ObjectSymmetries


def are_transforms_close(t1: pin.SE3, t2: pin.SE3) -> bool:
    """Checks if two SE3 transformations are close.

    :param t1: First transform to compare.
    :type t1: pinocchio.SE3
    :param t2: Second transform to compare.
    :type t2: pinocchio.SE3
    :return: If those transformations are close.
    :rtype: bool
    """
    diff = t1.inverse() * t2
    return np.linalg.norm(pin.log6(diff).vector) < 5e-3


def are_transform_msgs_close(t1: Transform, t2: Transform) -> bool:
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
    return are_transforms_close(T1, T2)


def is_transform_msg_in_list(t1: Transform, t_list: List[Transform]) -> bool:
    """Checks if a transform is in the list of transformations.

    :param t1: Transform to check if in the list.
    :type t1: Transform
    :param t_list: List of transforms to check if ``t1`` exists in there.
    :type t_list: List[Transform]
    :return: If the transform in the list.
    :rtype: bool
    """
    return any(are_transform_msgs_close(t1, t2) for t2 in t_list)


def is_transform_in_se3_list(
    t1: npt.NDArray[np.float64], t_list: List[pin.SE3]
) -> bool:
    """Checks if a transform is in the tensor of transformations.

    :param t1: Transform to check if in the tensor.
    :type t1: npt.NDArray[np.float64]
    :param t_list: List of SE3 objects used to find ``t1``.
    :type t_list: List[pinocchio.SE3]
    :return: If the transform in the list.
    :rtype: bool
    """
    T1 = pin.SE3(t1)
    return any(are_transforms_close(T1, t2) for t2 in t_list)


def pin_to_msg(transform: pin.SE3) -> Transform:
    """Converts 4x4 transformation matrix to ROS Transform message.

    :param transform: Input transformation.
    :type transform: pinocchio.SE3
    :return: Converted SE3 transformation into ROS Transform
        message format.
    :rtype: geometry_msgs.msg.Transform
    """
    pose_vec = pin.SE3ToXYZQUAT(transform)
    return Transform(
        translation=Vector3(**dict(zip("xyz", pose_vec[:3]))),
        rotation=Quaternion(**dict(zip("xyzw", pose_vec[3:]))),
    )


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


def test_only_discrete_np() -> None:
    t1 = pin.SE3(
        pin.Quaternion(np.array([0.0, 0.0, 0.0, 1.0])), np.array([0.0, 0.0, 0.0])
    )
    t2 = pin.SE3(
        pin.Quaternion(np.array([0.707, 0.0, 0.0, 0.707])), np.array([0.1, 0.1, 0.1])
    )
    msg = ObjectSymmetries(
        symmetries_discrete=[pin_to_msg(t1), pin_to_msg(t2)],
        symmetries_continuous=[],
    )

    res = discretize_symmetries(msg)

    assert res.shape == (2, 4, 4), "Result shape is incorrect!"

    for i, t in enumerate(res):
        assert is_transform_in_se3_list(t, [t1, t2]), (
            f"Discrete symmetry at index {i} did not match any of the initial ones!"
        )


def test_only_discrete_ros() -> None:
    msg = ObjectSymmetries(
        symmetries_discrete=[
            Transform(
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            Transform(
                translation=Vector3(x=0.1, y=0.1, z=0.1),
                rotation=Quaternion(x=0.707, y=0.0, z=0.0, w=0.707),
            ),
        ],
        symmetries_continuous=[],
    )

    res = discretize_symmetries(msg, return_ros_msg=True)

    assert len(res) == len(msg.symmetries_discrete), (
        "Results list does not have all discrete symmetries from message received!"
    )

    assert all(isinstance(r, Transform) for r in res), (
        "Returned type of elements in the list is not geometry_msgs.msg.Transform!"
    )

    for i, t in enumerate(res):
        assert is_transform_msg_in_list(t, msg.symmetries_discrete), (
            f"Discrete symmetry at index {i} did not match any initial ones in the message!"
        )


def test_only_continuous_np() -> None:
    n_symmetries = 2
    axis = np.array([1.0, 0.0, -1.0])
    axis = axis / np.linalg.norm(axis)
    offset = np.array([-0.1, 0.0, 0.1])

    t_list = [
        pin.SE3(np.eye(3), offset)
        * pin.SE3(
            pin.Quaternion(pin.AngleAxis(2.0 * np.pi / n_symmetries * i, axis)),
            np.array([0.0, 0.0, 0.0]),
        )
        * pin.SE3(np.eye(3), offset).inverse()
        for i in range(n_symmetries)
    ]
    msg = ObjectSymmetries(
        symmetries_discrete=[],
        symmetries_continuous=[
            ContinuousSymmetry(
                axis=Vector3(**dict(zip("xyz", axis))),
                offset=Vector3(**dict(zip("xyz", offset))),
            )
        ],
    )

    res = discretize_symmetries(msg, n_symmetries_continuous=n_symmetries)
    assert res.shape == (n_symmetries, 4, 4), "Result shape is incorrect!"
    for i, t in enumerate(res):
        assert is_transform_in_se3_list(t, t_list), (
            f"Discrete symmetry at index {i} did not match any symmetry from generated list!"
        )


def test_only_continuous_ros() -> None:
    n_symmetries = 9
    axis = np.array([1.0, 0.2, 1.0])
    axis = axis / np.linalg.norm(axis)
    offset = np.array([0.0, 0.0, 0.1])

    t_list = [
        pin.SE3(np.eye(3), offset)
        * pin.SE3(
            pin.Quaternion(pin.AngleAxis(2.0 * np.pi / n_symmetries * i, axis)),
            np.array([0.0, 0.0, 0.0]),
        )
        * pin.SE3(np.eye(3), offset).inverse()
        for i in range(n_symmetries)
    ]
    msg = ObjectSymmetries(
        symmetries_discrete=[],
        symmetries_continuous=[
            ContinuousSymmetry(
                axis=Vector3(**dict(zip("xyz", axis))),
                offset=Vector3(**dict(zip("xyz", offset))),
            )
        ],
    )

    res = discretize_symmetries(
        msg, n_symmetries_continuous=n_symmetries, return_ros_msg=True
    )

    assert len(res) == len(t_list), (
        "Results list does not have all discrete symmetries from message received!"
    )

    t_msgs = [pin_to_msg(t) for t in t_list]

    for i, t in enumerate(res):
        assert is_transform_msg_in_list(t, t_msgs), (
            f"Discrete symmetry at index {i} did not match any symmetry from generated list!"
        )


def test_mixed_np() -> None:
    n_symmetries = 32
    axis = np.array([-0.1, 0.5, 0.5])
    axis = axis / np.linalg.norm(axis)
    offset = np.array([-0.1, 0.6, 0.1])

    t_c_list = [
        pin.SE3(np.eye(3), offset)
        * pin.SE3(
            pin.Quaternion(pin.AngleAxis(2.0 * np.pi / n_symmetries * i, axis)),
            np.array([0.0, 0.0, 0.0]),
        )
        * pin.SE3(np.eye(3), offset).inverse()
        for i in range(n_symmetries)
    ]

    t_d_list = [
        pin.SE3(
            pin.Quaternion(np.array([0.0, 0.0, 0.0, 1.0])), np.array([0.0, 0.0, 0.0])
        ),
        pin.SE3(
            pin.Quaternion(np.array([0.707, 0.0, 0.0, 0.707])),
            np.array([0.1, 0.1, 0.1]),
        ),
    ]
    msg = ObjectSymmetries(
        symmetries_discrete=[*[pin_to_msg(t) for t in t_d_list]],
        symmetries_continuous=[
            ContinuousSymmetry(
                axis=Vector3(**dict(zip("xyz", axis))),
                offset=Vector3(**dict(zip("xyz", offset))),
            )
        ],
    )

    res = discretize_symmetries(msg, n_symmetries_continuous=n_symmetries)

    assert res.shape == (
        len(t_c_list) + len(t_d_list) + len(t_d_list) * len(t_c_list),
        4,
        4,
    ), "Result shape is incorrect!"

    t_test = [
        *t_c_list,
        *t_d_list,
        *[t_d * t_c for t_c in t_c_list for t_d in t_d_list],
    ]

    print(res, flush=True)

    for i, t in enumerate(res):
        assert is_transform_in_se3_list(t, t_test), (
            f"Discrete symmetry at index {i} did not match any symmetry from generated list!"
        )


def test_mixed_ros() -> None:
    n_symmetries = 31
    axis = np.array([-0.9, 0.2, -0.5])
    axis = axis / np.linalg.norm(axis)
    offset = np.array([-1.1, 0.6, 0.1])

    t_c_list = [
        pin.SE3(np.eye(3), offset)
        * pin.SE3(
            pin.Quaternion(pin.AngleAxis(2.0 * np.pi / n_symmetries * i, axis)),
            np.array([0.0, 0.0, 0.0]),
        )
        * pin.SE3(np.eye(3), offset).inverse()
        for i in range(n_symmetries)
    ]

    t_d_list = [
        pin.SE3(
            pin.Quaternion(np.array([0.0, 0.0, 0.0, 1.0])), np.array([0.0, 0.0, 0.0])
        ),
        pin.SE3(
            pin.Quaternion(np.array([0.707, 0.0, 0.0, 0.707])),
            np.array([0.1, 0.1, 0.1]),
        ),
    ]
    msg = ObjectSymmetries(
        symmetries_discrete=[*[pin_to_msg(t) for t in t_d_list]],
        symmetries_continuous=[
            ContinuousSymmetry(
                axis=Vector3(**dict(zip("xyz", axis))),
                offset=Vector3(**dict(zip("xyz", offset))),
            )
        ],
    )

    res = discretize_symmetries(
        msg, n_symmetries_continuous=n_symmetries, return_ros_msg=True
    )

    assert len(res) == (
        len(t_c_list) + len(t_d_list) + len(t_d_list) * len(t_c_list)
    ), "Size od the results is incorrect!"

    t_test = [
        *[pin_to_msg(t) for t in t_c_list],
        *[pin_to_msg(t) for t in t_d_list],
        *[pin_to_msg(t_d * t_c) for t_c in t_c_list for t_d in t_d_list],
    ]

    for i, t in enumerate(res):
        assert is_transform_msg_in_list(t, t_test), (
            f"Discrete symmetry at index {i} did not match any symmetry from generated list!"
        )
