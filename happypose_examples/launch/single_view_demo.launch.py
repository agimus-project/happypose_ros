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
    image_path = LaunchConfiguration("image_path")
    # Obtain value for FoV
    field_of_view = LaunchConfiguration("field_of_view")
    # Obtain URL of camera calibration. If FoV is non zero overwrite the calibration
    camera_info_url = (
        ""
        if float(field_of_view.perform(context)) > 0.0
        else LaunchConfiguration("camera_info_url")
    )

    # Start ROS node for image publishing
    image_publisher_node = Node(
        package="image_publisher",
        executable="image_publisher_node",
        namespace="cam_1",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "publish_rate": 23.0,
                "frame_id": "camera_1",
                "filename": image_path,
                "field_of_view": field_of_view,
                "camera_info_url": camera_info_url,
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
            "dataset_name": LaunchConfiguration("dataset_name"),
            "model_type": LaunchConfiguration("model_type"),
            "device": LaunchConfiguration("device"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "pose_estimator_type": LaunchConfiguration("pose_estimator_type")
        }.items(),
    )

    return [happypose_example_common_launch, image_publisher_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "dataset_name",
            default_value="ycbv",
            description="Name of BOP dataset, used to load specific weights and object models.",
        ),
        DeclareLaunchArgument(
            "model_type",
            default_value="pbr",
            description="Type of neural network model to use. Available: 'pbr'|'synth+real'.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cpu",
            description="Which device to load the models to.",
        ),
        DeclareLaunchArgument(
            "image_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("happypose_examples"),
                    "resources",
                    "000561.jpg",
                ]
            ),
            description="Path to image or webcam to be published as an input for happypose_ros node.",
        ),
        DeclareLaunchArgument(
            "field_of_view",
            default_value="0.0",
            description="Field of view of the camera taking images "
            + "used to approximate intrinsic parameters. "
            + "Overwrites `camera_info_url` parameter",
        ),
        DeclareLaunchArgument(
            "camera_info_url",
            default_value="package://happypose_examples/config/camera_info.yaml",
            description="URL of the calibrated camera params. Is overwritten by param `field_of_view`.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz with default view.",
        ),
        DeclareLaunchArgument(
            "pose_estimator_type",
            default_value="cosypose",
            description="Specifies which pose estimator to use in the pipeline.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
