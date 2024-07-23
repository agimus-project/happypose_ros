import numpy as np
import numpy.typing as npt
import transforms3d
from typing import List, Union

from geometry_msgs.msg import Transform, Vector3, Quaternion

from happypose_msgs.msg import ContinuousSymmetry, ObjectSymmetries


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
        return object_symmetries.symmetries_discrete

    def _discretize_continuous(
        sym: ContinuousSymmetry, idx: int
    ) -> npt.NDArray[np.float64]:
        axis = np.array([sym.axis.x, sym.axis.y, sym.axis.z])
        if not np.isclose(axis.sum(), 1.0):
            raise ValueError(
                f"Continuous symmetry at index {idx} has non unitary rotation axis!"
            )
        symmetries = np.zeros((n_symmetries_continuous, 4, 4))

        # Precompute steps of rotations
        rot_base = 2.0 * axis * np.pi / n_symmetries_continuous
        for i in range(n_symmetries_continuous):
            symmetries[i, :3, :3] = transforms3d.euler.euler2mat(*(rot_base * i))

        symmetries[:, -1, -1] = 1.0
        symmetries[:, :3, -1] = np.array([sym.offset.x, sym.offset.y, sym.offset.z])

        return symmetries

    symmetries_continuous = np.array(
        [
            _discretize_continuous(sym_c, idx)
            for idx, sym_c in enumerate(object_symmetries.symmetries_continuous)
        ]
    ).reshape((-1, 4, 4))

    def _transform_msg_to_mat(sym: Transform) -> npt.NDArray[np.float64]:
        M = np.eye(4)
        M[:3, :3] = transforms3d.quaternions.quat2mat(
            (sym.rotation.w, sym.rotation.x, sym.rotation.y, sym.rotation.z)
        )
        M[0, -1] = sym.translation.x
        M[1, -1] = sym.translation.y
        M[2, -1] = sym.translation.z
        return M

    symmetries_discrete = np.array(
        [
            _transform_msg_to_mat(sym_d)
            for sym_d in object_symmetries.symmetries_discrete
        ]
    ).reshape((-1, 4, 4))

    symmetries_mixed = [
        (symmetries_discrete[i] @ symmetries_continuous).reshape((-1, 4, 4))
        for i in range(len(object_symmetries.symmetries_discrete))
    ]

    symmetries = np.vstack(
        [symmetries_continuous, symmetries_discrete, *symmetries_mixed]
    )
    if not return_ros_msg:
        return symmetries

    def _mat_to_msg(M: npt.NDArray[np.float64]):
        q = transforms3d.quaternions.mat2quat(M[:3, :3])
        return Transform(
            translation=Vector3(**dict(zip("xyz", M[:, -1]))),
            rotation=Quaternion(**dict(zip("wxyz", q))),
        )

    return [_mat_to_msg(M) for M in symmetries]
