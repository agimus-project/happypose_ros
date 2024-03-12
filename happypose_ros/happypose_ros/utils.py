import os
import pathlib
import re

from ament_index_python.packages import get_package_share_directory


def unwrap_ros_path(path: str) -> pathlib.Path:
    out_path = path
    pattern_env = "\$\(env [A-Z_1-9]*\)"
    matches = re.findall(pattern_env, out_path)
    for match in matches:
        env_var = match[6:-1]
        if env_var not in os.environ:
            raise RuntimeError(f"No environment variable '{match}'.")
        env_path = os.environ[env_var]
        out_path = re.sub(pattern_env, env_path, out_path)

    pattern_package = "^package:\/\/[A-Za-z_1-9]*\/"
    matches = re.findall(pattern_package, out_path)
    if len(matches) > 0:
        package_path = get_package_share_directory(matches[0])
        out_path = re.sub(pattern_package, package_path, out_path)

    return pathlib.Path(out_path)
