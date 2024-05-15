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
    # Register image publishers in loop
    image_publishers = []
    for i in range(1, 4):
        # Obtain path to the image from ROS argument
        image_path = LaunchConfiguration(f"image_{i}_path")

        # Start ROS node for image publishing
        image_publishers.append(
            Node(
                package="image_publisher",
                executable="image_publisher_node",
                name=f"image_publisher_node_{i}",
                namespace=f"cam_{i}",
                parameters=[
                    {
                        "publish_rate": 10.0,
                        "frame_id": f"camera_{i}",
                        "filename": image_path,
                        # Camera info is ignored by the node on startup.
                        # Waiting for https://github.com/ros-perception/image_pipeline/issues/965
                        "camera_info_url": "package://happypose_examples/config/camera_info.yaml",
                    }
                ],
                remappings=[
                    # Remapped topics are created to match camera names in
                    # happypose_examples/config/cosypose_params_multiview.yaml
                    ("image_raw", f"/cam_{i}/image"),
                    ("camera_info", f"cam_{i}/camera_info"),
                ],
            )
        )

    # Evaluate path to happypose params
    happypose_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "config",
            "cosypose_params_multiview.yaml",
        ]
    )

    # Evaluate path to RViz config
    rviz_config_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "rviz",
            "happypose_multiview_example.rviz",
        ]
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
            "dataset_name": LaunchConfiguration("dataset_name"),
            "device": LaunchConfiguration("device"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "publish_camera_tf": "false",
            "rviz_config_path": rviz_config_path,
            "happypose_params_path": happypose_params_path,
        }.items(),
    )

    return [happypose_example_common_launch, *image_publishers]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "dataset_name",
            default_value="ycbv",
            description="Which dataset to use for inference.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cpu",
            description="Which device to load the models to.",
        ),
        DeclareLaunchArgument(
            "image_1_path",
            description="Path to the first image to be published "
            + "as an input for happypose_ros node.",
        ),
        DeclareLaunchArgument(
            "image_2_path",
            description="Path to the second image to be published "
            + "as an input for happypose_ros node.",
        ),
        DeclareLaunchArgument(
            "image_3_path",
            description="Path to the third image to be published "
            + "as an input for happypose_ros node.",
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
