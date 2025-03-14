from pathlib import Path
from typing import List

from setuptools import find_packages, setup

from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = "happypose_ros"
project_source_dir = Path(__file__).parent

module_name = "happypose_ros_parameters"
yaml_file = "happypose_ros/happypose_ros_parameters.yaml"
validation_module = "happypose_ros.custom_validation"
generate_parameter_module(module_name, yaml_file, validation_module=validation_module)


def get_files(dir: Path, pattern: str) -> List[str]:
    return [x.as_posix() for x in (dir).glob(pattern) if x.is_file()]


setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/happypose_node"]),
        ("share/happypose_ros", ["package.xml"]),
        (
            f"share/{package_name}/test",
            get_files(project_source_dir / "test", "*.py"),
        ),
        (
            f"share/{package_name}/test/rgb",
            get_files(project_source_dir / "test/rgb", "*.png"),
        ),
        (
            f"share/{package_name}/test",
            get_files(project_source_dir / "test", "*.png"),
        ),
        (
            f"share/{package_name}/test",
            get_files(project_source_dir / "test", "*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="ROS 2 wrapper around HappyPose python library for 6D pose estimation",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["happypose_node = happypose_ros.happypose_node:main"],
    },
)
