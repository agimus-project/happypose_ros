from setuptools import setup

package_name = "happypose_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="ROS 2 wrapper around Happypose python library for 6D pose estimation",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["happypose_node = happypose_ros.happypose_node:main"],
    },
)
