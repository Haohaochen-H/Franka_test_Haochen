# ROS1 VLM + YOLO integration

This folder is a copy of the original ROS1 Panda/RRT system with an added Python integration layer for YOLO and VLM planning.

Original ROS1 demo flow:

```text
franka_control.launch / demo_gazebo.launch
-> scripts/00_final_dynamic_demo.py
-> CameraOperations
-> PandaTransformations
-> RRT planner
-> MoveIt / FollowJointTrajectory / gripper
```

Added integration flow:

```text
scripts/vlm_yolo_dynamic_demo.py
-> CameraOperations captures RealSense color/depth
-> YoloObjectDetector returns structured detections
-> VlmPlanner generates a pick/place plan through Ollama
-> PlanGrounder maps object names to detected 3D points
-> optional RRTGroundedExecutor executes pick/place with the old RRT stack
```

## Files added

```text
catkin_ws/src/klemol_planner/klemol_planner/vlm_yolo/yolo_module.py
catkin_ws/src/klemol_planner/klemol_planner/vlm_yolo/vlm_module.py
catkin_ws/src/klemol_planner/klemol_planner/vlm_yolo/grounding_module.py
catkin_ws/src/klemol_planner/klemol_planner/vlm_yolo/yaw_estimator_adapter.py
catkin_ws/src/klemol_planner/scripts/vlm_yolo_dynamic_demo.py
catkin_ws/src/klemol_planner/models/best.pt
catkin_ws/src/klemol_planner/models/yolov8n.pt
external/YOLO_test/
```

## Expected setup

Start the old ROS1 robot stack first:

```bash
roslaunch panda_moveit_config franka_control.launch robot_ip:=172.16.0.3 load_gripper:=1
```

Then build/source this copied catkin workspace:

```bash
cd ros1_vlm_yolo_integration/catkin_ws
catkin_make
source devel/setup.bash
```

Install Python dependencies in the environment used by ROS1:

```bash
pip3 install -r ../external/YOLO_test/requirements.txt
```

Run a dry-run first. This performs camera calibration, YOLO detection, VLM planning, and grounding, but does not move the robot:

```bash
rosrun klemol_planner vlm_yolo_dynamic_demo.py \
  --instruction "pick the Cleaner_bottle and place it in the Salt_box"
```

Only after the grounded points look correct, execute:

```bash
rosrun klemol_planner vlm_yolo_dynamic_demo.py \
  --instruction "pick the Cleaner_bottle and place it in the Salt_box" \
  --execute
```

## Notes

- `--execute` moves the robot. Use the dry-run first.
- The script still depends on the old ArUco-based `PandaTransformations.calibrate_camera()`, so all required ArUco markers must be visible.
- The default YOLO weights path is `catkin_ws/src/klemol_planner/models/best.pt`. Pass `--weights path/to/model.pt` only when you want to override it.
- The VLM planner expects Ollama to be running at `http://localhost:11434` by default.

## Single-object YOLO pick test

For testing without VLM, use `single_test.py`. It detects one object, transforms the detection to the robot base frame, and optionally runs a conservative pick, lift, place-back sequence.

Dry-run:

```bash
rosrun klemol_planner single_test.py \
  --class-name Cleaner_bottle
```

Execute the pick-and-place-back after checking the printed base-frame point:

```bash
rosrun klemol_planner single_test.py \
  --class-name Cleaner_bottle \
  --execute
```

If `--class-name` is omitted, the script picks the highest-confidence detection.

To only pick and lift without placing the object back:

```bash
rosrun klemol_planner single_test.py \
  --class-name Cleaner_bottle \
  --execute \
  --skip-place
```

The scripts look for YOLO weights in this order:

```text
ros1_vlm_yolo_integration/catkin_ws/src/klemol_planner/models/best.pt
ros1_vlm_yolo_integration/catkin_ws/src/klemol_planner/models/yolov8n.pt
/home/haochenhe/YOLO_test/runs/detect/three_objects/weights/best.pt
ros1_vlm_yolo_integration/../YOLO_test/runs/detect/runs/detect/three_objects/weights/best.pt
ros1_vlm_yolo_integration/external/YOLO_test/runs/detect/three_objects/weights/best.pt
/home/haochenhe/YOLO_test/yolov8n.pt
ros1_vlm_yolo_integration/external/YOLO_test/yolov8n.pt
```

Use `--weights /path/to/model.pt` only when you want to override the default.
