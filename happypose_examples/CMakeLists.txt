cmake_minimum_required(VERSION 3.10)
project(happypose_examples)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  resources
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
