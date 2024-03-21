import numbers
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from typing import Any, Union

from rclpy.duration import Duration

from geometry_msgs.msg import Point, PoseWithCovariance, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros


def params_to_dict(params: happypose_ros.Params) -> dict:
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
    bbox = BoundingBox2D()
    if format == "xyxy":
        bbox.center.position.x = (bbox_data[0] + bbox_data[2]) / 2.0
        bbox.center.position.y = (bbox_data[1] + bbox_data[3]) / 2.0
        bbox.size_x = float(bbox_data[2] - bbox_data[0])
        bbox.size_y = float(bbox_data[3] - bbox_data[1])
    elif format == "xywh":
        bbox.center.position.x = (bbox_data[0] + bbox_data[2]) / 2.0
        bbox.center.position.y = (bbox_data[1] + bbox_data[3]) / 2.0
        bbox.size_x = float(bbox_data[2])
        bbox.size_y = float(bbox_data[3])
    else:
        raise ValueError(f"Unknown bounding box format: {format}")

    return bbox


def get_detection_array_msg(results: dict, header: Header) -> Detection2DArray:
    def generate_detection_msg(i: int) -> Detection2D:
        # Convert SE3 tensor to [x, y, z, qx, qy, qz, qw] pose representations
        pose_vec = pin.SE3ToXYZQUAT(pin.SE3(results["poses"][i].numpy()))
        detection = Detection2D(
            header=header,
            bbox=create_bounding_box_msg(results["bboxes"][i].numpy()),
            # Happypose supports only one result per detection, so the array
            # contains only single object
            results=[ObjectHypothesisWithPose()],
            # ID used for consistency across multiple detection messages.
            # Happypose does not differenciate between detected objects,
            # Hence emty string is used
            id="",
        )
        detection.results[0].hypothesis = ObjectHypothesis(
            class_id=results["infos"].label[i], score=results["infos"].score[i]
        )
        detection.results[0].pose = PoseWithCovariance(
            pose=Pose(
                position=Point(**dict(zip("xyz", pose_vec[:3]))),
                orientation=Quaternion(**dict(zip("xyzw", pose_vec[3:]))),
            ),
            # Happypose does not provide covariance, hence
            # it is hardcoded to identity matrix
            covariance=np.identity(6).reshape(-1).tolist(),
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
    label_to_strip: str = "",
    dynamic_opacity: bool = False,
) -> MarkerArray:
    def generate_marker_msg(i: int) -> Marker:
        detection = detections.detections[i]
        mesh_file_name = (
            detection.results[0].hypothesis.class_id.lstrip(label_to_strip)
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
            lifetime=Duration(seconds=10.0).to_msg(),
            pose=detection.results[0].pose.pose,
        )

    return MarkerArray(
        markers=[generate_marker_msg(i) for i in range(len(detections.detections))],
    )
