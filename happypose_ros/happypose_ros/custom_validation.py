import torch

from rclpy.parameter import Parameter


def torch_check_device(param: Parameter) -> str:
    if torch.cuda.is_available():
        available_devices = [f"cuda:{num}" for num in range(torch.cuda.device_count())]
    else:
        available_devices = []
    available_devices.append("cpu")
    if param.value in available_devices:
        return ""
    return (
        f"Device {param.name} is not available on this machine."
        f" Available devices: {available_devices}."
    )
