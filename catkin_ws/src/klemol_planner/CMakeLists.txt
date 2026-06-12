cmake_minimum_required(VERSION 3.0.2)
project(klemol_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

catkin_python_setup()
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install scripts (ROS executables)
catkin_install_python(PROGRAMS
  scripts/aruco_corner_debug.py
  scripts/demo_planner_node.py
  scripts/move_franka_klemol_based_on_camera.py
  scripts/single_test.py
  scripts/vlm_yolo_dynamic_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# Install Python modules from src/
install(DIRECTORY src/klemol_planner
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
