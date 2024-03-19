from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    # Obtain agument value for video device
    video_device = LaunchConfiguration("video_device")

    # Evaluate path of the cosypose paramteres
    happypose_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "config",
            "cosypose_params.yaml",
        ]
    )

    # Start ROS node of happypose
    happypose_node = Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        parameters=[happypose_params_path],
    )

    # Start ROS node for webcam feed
    webcam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        output="screen",
        name="usb_cam_node",
        parameters=[
            {
                "video_device": video_device,
                "framerate": 30.0,
                "io_method": "mmap",
                "frame_id": "webcam",
                "image_width": 640,
                "image_height": 480,
                "camera_name": "webcam",
                "camera_info_url": "package://happypose_examples/config/camera_info.yaml",
            }
        ],
    )

    return [happypose_node, webcam_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
