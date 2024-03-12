from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    happypose_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "config",
            "cosypose_params.yaml",
        ]
    )

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
        parameters=[happypose_params_path],
    )

    webcam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        output="screen",
        name="usb_cam_node",
        parameters=[webcam_params_path],
    )

    return LaunchDescription([happypose_node, webcam_node])
