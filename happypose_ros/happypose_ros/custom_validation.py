import torch

from rclpy.parameter import Parameter


def torch_check_device(param: Parameter) -> str:
    """Performs ROS parameter check to see if the selected Torch device
        is available on the machine.

    :param param: ROS parameter with a string containing the name of the Torch device.
    :type param: rclpy.Parameter
    :return: Error explanation. If empty string, everything is correct.
    :rtype: str
    """
    if torch.cuda.is_available():
        available_devices = [f"cuda:{num}" for num in range(torch.cuda.device_count())]
        available_devices.append("cuda")
    else:
        available_devices = []
    available_devices.append("cpu")
    if param.value in available_devices:
        return ""
    return (
        f"Device {param.name} is not available on this machine."
        f" Available devices: {available_devices}."
    )


def check_tf_valid_name(param: Parameter) -> str:
    """Checks if passed string can be used as a valid frame ID.

    :param param: ROS parameter with a string containing the name of frame ID.
    :type param: rclpy.Parameter
    :return: Error explanation. If empty string, everything is correct.
    :rtype: str
    """
    if param.value.startswith("/"):
        return (
            f"Invalid param '{param.name}' with value '{param.value}'. "
            "tf2 frame_ids cannot start with a '/'."
        )
    return ""


def check_teaserpp_installed(param: Parameter) -> str:
    """Checks if value of string is equal to ``teaserpp`` and tries to import the library.

    :param param: ROS parameter with a string containing the name of frame ID.
    :type param: rclpy.Parameter
    :return: Error explanation. If empty string, everything is correct.
    :rtype: str
    """
    if param.value == "teaserpp":
        try:
            from happypose.pose_estimators.megapose.inference.teaserpp_refiner import (  # noqa: F401
                TeaserppRefiner,
            )
        except Exception:
            return (
                f"Can't use '{param.name}' with value '{param.value}'. "
                "`TeaserppRefiner` was not installed in the system."
            )
    return ""
