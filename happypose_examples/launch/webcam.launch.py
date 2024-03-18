from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    happypose_params_path = PathJoinSubstitution(
        [
            FindPackageShare("happypose_examples"),
            "config",
            "cosypose_params.yaml",
        ]
    )

    happypose_node = Node(
        # prefix="python3 -m pdb",
        package="happypose_ros",
        executable="happypose_node",
        name="happypose_node",
        parameters=[happypose_params_path],
    )

    # webcam_node = Node(
    #     package="usb_cam",
    #     executable="usb_cam_node_exe",
    #     output="screen",
    #     name="usb_cam_node",
    #     parameters=[
    #         {
    #             "video_device": "/dev/video0",
    #             "framerate": 30.0,
    #             "io_method": "mmap",
    #             "frame_id": "webcam",
    #             "image_width": 640,
    #             "image_height": 480,
    #             "camera_name": "webcam",
    #             "camera_info_url": "package://happypose_examples/config/camera_info.yaml",
    #         }
    #     ],
    # )

    return LaunchDescription([happypose_node])
