from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_context import LaunchContext
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    # Obtain argument specifying if RViz should be launched
    use_rviz = LaunchConfiguration("use_rviz")

    # Obtain argument specifying path from which to load happypose_ros parameters
    happypose_params_path = LaunchConfiguration("happypose_params_path")

    # Obtain argument specifying path from which to load RViz config
    rviz_config_path = LaunchConfiguration("rviz_config_path")

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
        happypose_node,
        rviz_node,
        static_transform_publisher_node,
    ]


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
            "happypose_params_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("happypose_examples"),
                    "config",
                    "cosypose_params.yaml",
                ]
            ),
            description="Path to a file containing happypose_ros node parameters.",
        ),
        DeclareLaunchArgument(
            "publish_camera_tf",
            default_value="true",
            description="Publish static transforamtion for the camera.",
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
            "use_rviz",
            default_value="false",
            description="Launch RViz with default view.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
