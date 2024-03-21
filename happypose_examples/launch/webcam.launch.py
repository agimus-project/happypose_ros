from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_context import LaunchContext
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    # Obtain agument value for video device
    video_device = LaunchConfiguration("video_device")

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
                "frame_id": "camera",
                "image_width": 640,
                "image_height": 480,
                "camera_name": "webcam",
                "camera_info_url": "package://happypose_examples/config/camera_info.yaml",
            }
        ],
    )

    # Include common part of the demo launch files
    happypose_example_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("happypose_examples"),
                        "launch",
                        "common.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items(),
    )

    return [happypose_example_common_launch, webcam_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "video_device",
            default_value="/dev/video0",
            description="Device name of a video device to publish "
            + "as a camera feed for happypose_ros node",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz with default view.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
