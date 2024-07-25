from copy import copy
import numpy as np
import numpy.typing as npt
import transforms3d
from typing import List, Union

from geometry_msgs.msg import Transform, Vector3, Quaternion

from happypose_msgs.msg import ObjectSymmetries


def discretize_symmetries(
    object_symmetries: ObjectSymmetries,
    n_symmetries_continuous: int = 8,
    return_ros_msg: bool = False,
) -> Union[npt.NDArray[np.float64], List[Transform]]:
    """Converts discrete and continuous symmetries to a list of discrete symmetries.

    :param object_symmetries: ROS message containing symmetries of a given object.
    :type object_symmetries: happypose_msgs.msg.ObjectSymmetries
    :param n_symmetries_continuous: Number of segments to discretize continuous symmetries.
    :type n_symmetries_continuous: int
    :param return_ros_msg: Whether to return ROS message or numpy array
        with 4x4 matrices, defaults to False.
    :type return_ros_msg: bool, optional
    :return: If ``return_ros_msg`` is False returns array of a shape (n, 4, 4) with ``n``
        SE3 transformation matrices representing symmetries.
        Otherwise list of ROS Transform messages.
    :rtype: Union[npt.NDArray[np.float64], List[geometry_msgs.msg.Transform]]
    """

    # If there are not continuous symmetries and ROS message is expected skip computations
    if return_ros_msg and len(object_symmetries.symmetries_continuous) == 0:
        return copy(object_symmetries.symmetries_discrete)

    n_con = len(object_symmetries.symmetries_continuous) * n_symmetries_continuous
    n_disc = len(object_symmetries.symmetries_discrete)
    n_mix = n_con * n_disc

    # Preallocate memory for results
    out = np.zeros((n_con + n_disc + n_mix, 4, 4))

    # Precompute steps of rotations
    rot_base = 2.0 * np.pi / n_symmetries_continuous

    # Discretize continuous symmetries
    for i, sym_c in enumerate(object_symmetries.symmetries_continuous):
        axis = np.array([sym_c.axis.x, sym_c.axis.y, sym_c.axis.z])
        if not np.isclose(np.linalg.norm(axis), 1.0):
            raise ValueError(
                f"Continuous symmetry at index {i} has non unitary rotation axis!"
            )
        # Compute begin and end indices
        begin = i * n_symmetries_continuous
        end = (i + 1) * n_symmetries_continuous
        out[begin:end, :3, :3] = np.array(
            [
                transforms3d.axangles.axangle2mat(axis, rot_base * j)
                for j in range(n_symmetries_continuous)
            ]
        )
        out[begin:end, :, -1] = np.array(
            [sym_c.offset.x, sym_c.offset.y, sym_c.offset.z, 1.0]
        )

    # Convert discrete symmetries to matrix format
    for i, sym_d in enumerate(object_symmetries.symmetries_discrete):
        begin = n_con + i
        out[begin, :3, :3] = transforms3d.quaternions.quat2mat(
            [sym_d.rotation.w, sym_d.rotation.x, sym_d.rotation.y, sym_d.rotation.z]
        )
        out[begin, :, -1] = np.array(
            [sym_d.translation.x, sym_d.translation.y, sym_d.translation.z, 1.0]
        )

    sym_c_d_end = n_con + n_disc
    symmetries_continuous = out[:n_con]
    # Combine discrete symmetries with possible continuous rotations
    # TODO @MedericFourmy we should ensure this operation is valid for all object
    # and not only objects with offset being at the origin of the coordinate system.
    for i in range(n_disc):
        begin = sym_c_d_end + i * n_symmetries_continuous
        end = sym_c_d_end + (i + 1) * n_symmetries_continuous
        symmetry_discrete = out[n_con + i]
        # Multiply batch of continuous symmetries onto single discrete symmetry
        out[begin:end] = symmetry_discrete @ symmetries_continuous

    if not return_ros_msg:
        return out

    def _mat_to_msg(M: npt.NDArray[np.float64]) -> Transform:
        q = transforms3d.quaternions.mat2quat(M[:3, :3])
        return Transform(
            translation=Vector3(**dict(zip("xyz", M[:, -1]))),
            rotation=Quaternion(**dict(zip("wxyz", q))),
        )

    return [_mat_to_msg(M) for M in out]
