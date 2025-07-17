from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)

from launch.conditions import IfCondition
from launch.launch_context import LaunchContext
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


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

    # Obtain argument specifying if RViz should be launched
    use_rviz = LaunchConfiguration("use_rviz")

    # Obtain argument specifying path from which to load happypose_ros parameters
    happypose_params_path = LaunchConfiguration("happypose_params_path")

    # Obtain argument specifying path from which to load RViz config
    rviz_config_path = LaunchConfiguration("rviz_config_path")

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

    # Start ROS node of happypose
    happypose_node = Node(
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        parameters=[ParameterFile(param_file=happypose_params_path, allow_substs=True)],
    )

    # Start RViz2 ROS node
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
    )

    # Start static TF publisher to transform
    # camera optical frame and rotate it for better rviz preview
    static_transform_publisher_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        arguments=[
            "--roll",
            "-1.57",
            "--yaw",
            "-1.57",
            "--frame-id",
            "world",
            "--child-frame-id",
            "camera_1",
        ],
    )

    return [
        image_publisher_node,
        happypose_node,
        rviz_node,
        static_transform_publisher_node,
    ]


def generate_launch_description():
    declared_arguments = [
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
            default_value="package://happypose_examples/config/camera_info_megapose.yaml",
            description="URL of the calibrated camera params. Is overwritten by param `field_of_view`.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz with default view.",
        ),
        DeclareLaunchArgument(
            "pose_estimator_type",
            default_value="megapose",
            description="Specifies which pose estimator to use in the pipeline.",
        ),
        DeclareLaunchArgument(
            "rviz_config_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("happypose_examples"),
                    "rviz",
                    "happypose_example.rviz",
                ]
            ),
            description="Path to a file containing RViz view configuration.",
        ),
        DeclareLaunchArgument(
            "publish_camera_tf",
            default_value="true",
            description="Publish static transformation for the camera.",
        ),
        DeclareLaunchArgument(
            "happypose_params_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("happypose_examples"),
                    "config",
                    "megapose_params.yaml",
                ]
            ),
            description="Path to a file containing happypose_ros node parameters.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
