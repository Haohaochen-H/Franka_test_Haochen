cmake_minimum_required(VERSION 3.0.2)
project(klemol_planner)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  geometry_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  GenerateVlmPlan.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs geometry_msgs
)

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
  scripts/object_hover_plane_debug.py
  scripts/object_perception_debug.py
  scripts/single_test.py
  scripts/yolo_pose_debug.py
  scripts/vlm_yolo_dynamic_demo.py
  scripts/vlm_planner_node.py
  scripts/vlm_plan_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# Install Python modules from src/
install(DIRECTORY src/klemol_planner
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
