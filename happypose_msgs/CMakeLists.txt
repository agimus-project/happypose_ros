cmake_minimum_required(VERSION 3.10.2)
project(happypose_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/BlockingPipelieTrigger.srv"
  DEPENDENCIES vision_msgs
)

ament_python_install_package(utils)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
