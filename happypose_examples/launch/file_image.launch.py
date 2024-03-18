from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # happypose_params_path = PathJoinSubstitution(
    #     [
    #         FindPackageShare("happypose_examples"),
    #         "config",
    #         "cosypose_params.yaml",
    #     ]
    # )
    # happypose_node = Node(
    #     package="happypose_ros",
    #     executable="happypose_node",
    #     name="happypose_node",
    #     parameters=[happypose_params_path],
    # )

    image_file_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "resource",
            "000071.png",
        ]
    )
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

    return LaunchDescription([image_publisher_node])
