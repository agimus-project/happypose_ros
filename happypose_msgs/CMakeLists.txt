cmake_minimum_required(VERSION 3.10)
project(happypose_msgs)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(rclpy REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  msg/ContinuousSymmetry.msg
  msg/ObjectSymmetries.msg
  msg/ObjectSymmetriesArray.msg
  DEPENDENCIES
  builtin_interfaces
  std_msgs
  geometry_msgs)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}_py)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
