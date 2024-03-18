import numbers

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from rclpy.duration import Duration

from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

from happypose_ros.happypose_ros_parameters import happypose_ros


def params2dict(params: happypose_ros.Params) -> dict:
    out = {}

    def to_dict_internal(instance, name, base_dict):
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


def pose2marker(pose: PoseStamped, label: str, idx: int) -> Marker:
    mesh_path = LOCAL_DATA_DIR / f"bop_datasets/ycbv/models/{label}.ply"

    return Marker(
        id=idx,
        mesh_resource="file://" + mesh_path.as_posix(),
        mesh_use_embedded_materials=True,
        type=Marker.MESH_RESOURCE,
        header=pose.header,
        scale=Vector3(**dict(zip("xyz", [0.001] * 3))),
        color=ColorRGBA(**dict(zip("rgba", [1.0] * 4))),
        lifetime=Duration(seconds=10.0).to_msg(),
        pose=pose.pose,
    )
