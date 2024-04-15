from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_context import LaunchContext
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    # Obtain argument specifying if RViz should be launched
    use_rviz = LaunchConfiguration("use_rviz")

    # Evaluate path of the cosypose parameters
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

    # Evaluate path of the cosypose parameters
    rviz_config_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "rviz",
            "happypose_example.rviz",
        ]
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
        condition=IfCondition(use_rviz),
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
            "camera",
        ],
    )

    return [
        happypose_node,
        rviz_node,
        static_transform_publisher_node,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz with default view.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
