import numbers
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from typing import Any, Union
from torch import Tensor

from rclpy.duration import Duration

from geometry_msgs.msg import (
    Point,
    PoseWithCovariance,
    Pose,
    Transform,
    TransformStamped,
    Quaternion,
    Vector3,
)
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from happypose_msgs.msg import (
    ContinuousSymmetry,
    ObjectSymmetries,
    ObjectSymmetriesArray,
)

import happypose.toolbox.lib3d.symmetries as happypose_symmetries
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros


def params_to_dict(params: happypose_ros.Params) -> dict:
    """Converts an object created by generate_parameter_library to a Python dictionary.
    Parameters are converted from 'my_params.foo.bar' to 'my_params["foo"]["bar"]'.

    :param params: Object created by generate_parameter_library with ROS parameters.
    :type params: happypose_ros.Params
    :return: ROS parameters converted to a dictionary.
    :rtype: dict
    """
    out = {}

    def to_dict_internal(instance: Any, name: str, base_dict: dict) -> None:
        if isinstance(instance, (str, numbers.Number, list)):
            base_dict.update({name: instance})
        else:
            if name != "":
                base_dict.update({name: {}})
            data = [
                attr
                for attr in dir(instance)
                if (
                    not callable(getattr(instance, attr))
                    and not attr.startswith("__")
                    and attr != "stamp_"
                )
            ]
            for attr in data:
                to_dict_internal(
                    getattr(instance, attr),
                    attr,
                    base_dict[name] if name != "" else base_dict,
                )

    to_dict_internal(params, "", out)
    return out


def create_bounding_box_msg(
    bbox_data: Union[list[float], npt.NDArray[np.float32]], format: str = "xyxy"
) -> BoundingBox2D:
    """Converts a bounding box passed as a list or array to a ROS message.

    :param bbox_data: List of key points in the bounding box.
    :type bbox_data: Union[list[float], numpy.typing.NDArray[numpy.float32]]
    :param format: Format in which the bounding box is stored, allowed "xyxy" and "xywh", defaults to "xyxy".
    :type format: str, optional
    :raises ValueError: Incorrect value of :param: format.
    :return: Bounding box converted to a ROS message.
    :rtype: vision_msgs.msg.BoundingBox2D
    """
    bbox = BoundingBox2D()
    if format == "xyxy":
        bbox.center.position.x = float(bbox_data[0] + bbox_data[2]) / 2.0
        bbox.center.position.y = float(bbox_data[1] + bbox_data[3]) / 2.0
        bbox.size_x = float(bbox_data[2] - bbox_data[0])
        bbox.size_y = float(bbox_data[3] - bbox_data[1])
    elif format == "xywh":
        bbox.center.position.x = float(bbox_data[0] + (bbox_data[2] / 2.0))
        bbox.center.position.y = float(bbox_data[1] + (bbox_data[3] / 2.0))
        bbox.size_x = float(bbox_data[2])
        bbox.size_y = float(bbox_data[3])
    else:
        raise ValueError(f"Unknown bounding box format: {format}")

    return bbox


def get_detection_array_msg(
    results: dict, header: Header, has_bbox: bool = True
) -> Detection2DArray:
    """Converts results dictionary to a Detection2DArray message.

    :param results: Detections obtained from HappyPose pipeline.
    :type results: dict
    :param header: Header to pass to the message.
    :type header: std_msgs.msg.Header
    :param has_bbox: Indicates if bounding box has to be populated
        or left empty, defaults to True.
    :type has_bbox: bool, optional
    :return: ROS message with detection.
    :rtype: vision_msgs.msg.Detection2DArray
    """

    def generate_detection_msg(i: int) -> Detection2D:
        # Convert SE3 tensor to [x, y, z, qx, qy, qz, qw] pose representations
        pose_vec = pin.SE3ToXYZQUAT(pin.SE3(results["poses"][i].numpy()))
        detection = Detection2D(
            header=header,
            # HappyPose supports only one result per detection, so the array
            # contains only single object
            results=[ObjectHypothesisWithPose()],
            # ID is used for consistency across multiple detection messages.
            # HappyPose does not differentiate between detected objects,
            # Hence empty string is used.
            id="",
        )
        if has_bbox:
            detection.bbox = create_bounding_box_msg(results["bboxes"][i].numpy())
        detection.results[0].hypothesis = ObjectHypothesis(
            class_id=results["infos"].label[i], score=results["infos"].score[i]
        )
        detection.results[0].pose = PoseWithCovariance(
            pose=Pose(
                position=Point(**dict(zip("xyz", pose_vec[:3]))),
                orientation=Quaternion(**dict(zip("xyzw", pose_vec[3:]))),
            ),
            # HappyPose does not provide covariance, hence
            # it is hard-coded to identity matrix
            covariance=[1.0 if (i % 7) == 0 else 0.0 for i in range(36)],
        )
        return detection

    return Detection2DArray(
        header=header,
        detections=[generate_detection_msg(i) for i in range(len(results["infos"]))],
    )


def get_marker_array_msg(
    detections: Detection2DArray,
    mesh_folder_url: str,
    mesh_file_extension: str = "ply",
    prefix: str = "",
    dynamic_opacity: bool = False,
    marker_lifetime: float = 10.0,
) -> MarkerArray:
    """Converts Detection2DArray to MarkerArray ROS message for visualization.

    :param detections: Detections messages to get information from.
    :type detections: vision_msgs.msg.Detection2DArray
    :param mesh_folder_url: Path from which meshes will be later fetched for visualization.
    :type mesh_folder_url: str
    :param mesh_file_extension: Extension file format of the meshes, defaults to "ply".
    :type mesh_file_extension: str, optional
    :param prefix: Prefix used to subtract from the object class name, used to create valid mesh paths, defaults to "".
    :type prefix: str, optional
    :param dynamic_opacity: Whether to use detection score as opacity of a mesh, defaults to False.
    :type dynamic_opacity: bool, optional
    :param marker_lifetime: Value to set as a lifetime of the marker in seconds, defaults to 10.0.
    :type marker_lifetime: float, optional
    :return: Message ready to publish for visualization.
    :rtype: visualization_msgs.msg.MarkerArray
    """

    def generate_marker_msg(i: int) -> Marker:
        detection = detections.detections[i]
        mesh_file_name = (
            detection.results[0].hypothesis.class_id.removeprefix(prefix)
            + "."
            + mesh_file_extension
        )
        return Marker(
            id=i,
            mesh_resource=f"{mesh_folder_url}/{mesh_file_name}",
            mesh_use_embedded_materials=True,
            type=Marker.MESH_RESOURCE,
            header=detection.header,
            scale=Vector3(**dict(zip("xyz", [0.001] * 3))),
            color=ColorRGBA(
                **dict(zip("rgb", [1.0] * 3)),
                a=detection.results[0].hypothesis.score if dynamic_opacity else 1.0,
            ),
            lifetime=Duration(seconds=marker_lifetime).to_msg(),
            pose=detection.results[0].pose.pose,
        )

    return MarkerArray(
        markers=[generate_marker_msg(i) for i in range(len(detections.detections))],
    )


def transform_mat_to_msg(transform: npt.NDArray[np.float64]) -> Transform:
    """Converts 4x4 transformation matrix to ROS Transform message.

    :param transform: 4x4 transformation array.
    :type transform: npt.NDArray[np.float64]
    :return: Converted SE3 transformation into ROS Transform
        message format.
    :rtype: geometry_msgs.msg.Transform
    """
    pose_vec = pin.SE3ToXYZQUAT(pin.SE3(transform))
    return Transform(
        translation=Vector3(**dict(zip("xyz", pose_vec[:3]))),
        rotation=Quaternion(**dict(zip("xyzw", pose_vec[3:]))),
    )


def get_camera_transform(
    camera_pose: Tensor, header: Header, child_frame_id: str
) -> TransformStamped:
    """Convert SE3 tensor from HappyPose to ROS Transformation message.

    :param camera_pose: SE3 tensor.
    :type camera_pose: torch.Tensor
    :param header: Header used to populate the message.
        Contains frame id of leading camera.
    :type header: std_msgs.msg.Header
    :param child_frame_id: Frame id of the camera which TF will be published.
    :type child_frame_id: str
    :return: ROS message representing transformation between
        leading camera and estimated camera pose.
    :rtype: geometry_msgs.msg.TransformStamped
    """

    return TransformStamped(
        header=header,
        child_frame_id=child_frame_id,
        transform=transform_mat_to_msg(camera_pose.numpy()),
    )


def continuous_symmetry_to_msg(
    symmetry: happypose_symmetries.ContinuousSymmetry,
) -> ContinuousSymmetry:
    """Converts HappyPose ContinuousSymmetry object into ContinuousSymmetry ROS message

    :param symmetry: HappyPose object storing definition of continuous symmetry.
    :type symmetry: happypose_symmetries.ContinuousSymmetry
    :return: ROS message representing continuous symmetry.
    :rtype: happypose_msgs.msg.ContinuousSymmetry
    """
    return ContinuousSymmetry(
        # Explicit conversion to float is required in this case
        offset=Vector3(**dict(zip("xyz", [float(i) for i in symmetry.offset]))),
        axis=Vector3(**dict(zip("xyz", [float(i) for i in symmetry.axis]))),
    )


def discrete_symmetry_to_msg(
    symmetry: happypose_symmetries.DiscreteSymmetry,
) -> Transform:
    """Converts HappyPose DiscreteSymmetry object into Transform ROS message corresponding to

    :param symmetry: HappyPose object storing definition of continuous symmetry.
    :type symmetry: happypose_symmetries.DiscreteSymmetry
    :return: ROS message with transformation corresponding to given symmetry.
    :rtype: geometry_msgs.msg.Transform
    """
    return transform_mat_to_msg(symmetry.pose)


def get_object_symmetries_msg(
    dataset: RigidObjectDataset, header: Header
) -> ObjectSymmetriesArray:
    """Converts HappyPose RigidObjectDataset object into ROS message representing symmetries
    of all objects in that dataset.

    :param dataset: Dataset of rigid objects detected by HappyPose.
    :type dataset: RigidObjectDataset
    :param header: Header used to populate the message.
        Contains timestamp at which message was published.
    :type header: std_msgs.msg.Header
    :return: Message containing symmetries of objects detected by HappyPose.
    :rtype: happypose_msgs.msg.ObjectSymmetriesArray
    """

    def generate_symmetry_msg(object: RigidObject) -> ObjectSymmetries:
        return ObjectSymmetries(
            class_id=object.label,
            symmetries_discrete=[
                discrete_symmetry_to_msg(sym) for sym in object.symmetries_discrete
            ],
            symmetries_continuous=[
                continuous_symmetry_to_msg(sym) for sym in object.symmetries_continuous
            ],
        )

    return ObjectSymmetriesArray(
        header=header,
        objects=[generate_symmetry_msg(object) for object in dataset.list_objects],
    )
