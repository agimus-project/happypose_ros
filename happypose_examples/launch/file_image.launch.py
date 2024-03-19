from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    # Obtain argument value for image path
    image_file_path = LaunchConfiguration("image_file_path")

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
                "camera_info_url": "package://happypose_examples/config/camera_info.yaml",
            }
        ],
    )

    return [happypose_node, image_publisher_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("image_file_path", default_value=""),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
