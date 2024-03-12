from happypose_ros.utils import unwrap_ros_path


def validate_path_folder(param) -> str:
    try:
        path = unwrap_ros_path(param.value)
    except RuntimeError as e:
        return (
            f"Failed to parse param '{param.name}' with path '{param.value}'. {str(e)}"
        )

    if not path.exists():
        return f"Path '{path.as_posix}' from paramter '{param.name}' does not exist!"
    return ""
