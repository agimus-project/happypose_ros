import os
import re
import uuid

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    happypose_params_path = (
        get_package_share_directory("happypose_examples")
        + "/config/cosypose_params.yaml"
    )
    # Replace all "$(env HAPPYPOSE_DATA_DIR)" with actual content of the env variable
    tmp_cosypose_params_file = f"/tmp/{str(uuid.uuid4())}_cosypose_params.yaml"
    with open(happypose_params_path, "r") as input_file:
        with open(tmp_cosypose_params_file, "w") as output_file:
            cosypose_params = input_file.read()
            pattern = "\$\(env [A-Z_1-9]*\)"
            matches = re.findall(pattern, cosypose_params)
            for match in matches:
                env_path = os.environ[match[6:-1]]
                cosypose_params = re.sub(pattern, env_path, cosypose_params)
            output_file.write(cosypose_params)

    webcam_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "config",
            "webcam_params.yaml",
        ]
    )

    happypose_node = Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        parameters=[tmp_cosypose_params_file],
    )

    webcam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        output="screen",
        name="usb_cam_node",
        parameters=[webcam_params_path],
    )

    return LaunchDescription([happypose_node, webcam_node])
