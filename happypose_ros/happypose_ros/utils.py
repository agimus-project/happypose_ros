import numbers
from typing import Any

from rclpy.duration import Duration

from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


# Automatically generated file
from happypose_ros.happypose_ros_parameters import happypose_ros

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
)


def params2dict(params: happypose_ros.Params) -> dict:
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


def detection2darray_msg_to_marker_array_msg(
    detections: Detection2DArray,
    mesh_folder_url: str,
    mesh_file_extension: str = "ply",
    dynamic_opacity: bool = False,
) -> Marker:
    markers = [None] * len(detections.detections)
    for i, detection in enumerate(detections):
        detection = Detection2D()
        markers[i] = Marker(
            id=i,
            mesh_resource=f"{mesh_folder_url}/{detection.id}.{mesh_file_extension}",
            mesh_use_embedded_materials=True,
            type=Marker.MESH_RESOURCE,
            header=detection.header,
            scale=Vector3(**dict(zip("xyz", [0.001] * 3))),
            color=ColorRGBA(
                **dict(zip("rgb", [1.0] * 3)),
                a=detection.results[0].hypothesis.score if dynamic_opacity else 1.0,
            ),
            lifetime=Duration(seconds=10.0).to_msg(),
            pose=detection.results[0].pose,
        )
    return markers


# def tensor_collection_to_detection2darray_msg(
#     results: TensorCollection, header: Header
# ) -> Detection2DArray:
#     detections = [None] * len(results.infos)
#     for i in range(len(results.infos)):
#         pose_vec = pin.SE3ToXYZQUAT(pin.SE3(results.poses[i].numpy()))
