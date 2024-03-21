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
    # Obtain argument value for image path
    image_file_path = LaunchConfiguration("image_file_path")

    # Start ROS node for image publishing
    image_publisher_node = Node(
        package="image_publisher",
        executable="image_publisher_node",
        output="screen",
        arguments=[image_file_path],
        parameters=[
            {
                "use_sim_time": False,
                "publish_rate": 10.0,
                # Ignored by the node, fixed by https://github.com/ros-perception/image_pipeline/pull/861
                # Currently no bug fix for humble, requires back port
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

    return [happypose_example_common_launch, image_publisher_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "image_file_path",
            description="Path to image to be published as an input for happypose_ros node.",
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
