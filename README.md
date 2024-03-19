# happypose_ros
ROS 2 wrapper for a 6D pose estimation library, [Happypose](https://github.com/agimus-project/happypose).

## Build instructions


:warning: Conda installation is not supported

:warning: GPU is currently not supported!

Currently, there is no automated build for happypose library itself built into the ROS node. Please follow the installation guide found in the [happypose README.md](https://github.com/agimus-project/happypose?tab=readme-ov-file#example-with-venv).

```bash
vcs import --recursive < src/happypose_ros/happypose_ros.repos
rosdep update --rosdistro $ROS_DISTRO
rosdep install -y -i --from-paths src --rosdistro $ROS_DISTRO
# paramter --symlink-install is optional
catkin build --symlink-install
```

## Launch

:information: Intrinsic parameters of the camera are approximate in the demos and may cause inaccurate results! You can change them by modifying [camera_info.yaml](./happypose_examples/config/camera_info.yaml) file.

To launch with webcam preview run:
```bash
ros2 launch happypose_examples webcam.launch.py video_device:="/dev/video0"
```
This will subscribe to `/dev/video0` input and feed this image to the **happypose_ros** node.


Another option it to stream static image from a file:
```bash
ros2 launch happypose_examples file_image.launch.launch.py image_file_path:=<path to the image>
```
