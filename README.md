# happypose_ros
ROS 2 wrapper for a 6D pose estimation library, [Happypose](https://github.com/agimus-project/happypose).

## Build instructions


:warning: Conda installation is not supported

Currently, there is no automated build for happypose library itself built into the ROS node. Please follow the installation guide found in the [happypose README.md](https://github.com/agimus-project/happypose?tab=readme-ov-file#example-with-venv).

```bash
rosdep update --rosdistro $ROS_DISTRO
rosdep install -y -i --from-paths src --rosdistro $ROS_DISTRO
# parameter --symlink-install is optional
colcon build --symlink-install
```

## Launch

:warning: Intrinsic parameters of the camera are approximate in the demos and may cause inaccurate results! You can change them by modifying the `k_matrix` param in [cosypose_params.yaml](./happypose_examples/config/cosypose_params.yaml) file.

To launch the demo run:
```bash
ros2 launch happypose_examples single_view_demo.launch.py use_rviz:=true device:=cpu dataset_name:=ycbv \
    image_path:=<path to the image>
```
The `<path to the image>` can either be a relative or global path to an image stored on the computer's drive or a path to a video device mapped to a webcam, e.g. `/dev/video0`.

To evaluate multiview capabilities try:
```bash
ros2 launch happypose_examples multi_view_demo.launch.py use_rviz:=true device:=cpu dataset_name:=ycbv \
    image_1_path:=<path to the image> \
    image_2_path:=<path to the image> \
    image_3_path:=<path to the image>
```
