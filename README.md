# Reto Final Puzzlebot — Autonomous Navigation with Vision for Puzzlebot

> **ROS 2 (Humble)** workspace for a differential-drive Puzzlebot that follows a line, obeys traffic lights, detects pedestrian crosswalks, and reacts to road signs — all on a Jetson board using a custom TensorRT YOLOv4-tiny model.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [System Architecture](#system-architecture)
4. [Packages](#packages)
   - [vision\_puzzlebot](#vision_puzzlebot)
   - [vision\_puzzlebot\_trt](#vision_puzzlebot_trt)
   - [waypoints\_puzzlebot](#waypoints_puzzlebot)
5. [ROS 2 Topics](#ros-2-topics)
6. [State Machine](#state-machine)
7. [Detected Signs & Classes](#detected-signs--classes)
8. [Configuration](#configuration)
9. [Launch Files](#launch-files)
10. [Dependencies](#dependencies)
11. [Build & Run](#build--run)

---

## Overview

Reto_Final_Puzzlebot implements a full autonomous-driving pipeline for the Puzzlebot platform. The robot:

- Follows a dark line on the floor using Otsu thresholding and contour analysis.
- Detects **pedestrian crosswalks** by looking for clusters of six horizontally-aligned contours and aligns to their angle.
- Classifies **traffic lights** (red / yellow / green) via HSV colour segmentation.
- Recognises six **road signs** in real time with a YOLOv4-tiny model accelerated with NVIDIA TensorRT (FP16).
- Arbitrates all perception inputs through a **state machine** that issues wheel-velocity commands.

---

## Repository Structure

```
Reto_Final_Puzzlebot/
└── src/
    ├── vision_puzzlebot/          # Python pkg — camera, line follower, traffic-light detector
    │   ├── config/
    │   │   ├── params.yaml
    │   │   └── crosswalk_debug_conf.yaml
    │   ├── launch/
    │   │   ├── final_launch.launch.py
    │   │   └── puzzlebot_line_follower.launch.py
    │   └── vision_puzzlebot/
    │       ├── cam_publish.py
    │       ├── line_follower_camera.py
    │       └── trafficlight_detect.py
    ├── vision_puzzlebot_trt/      # C++ pkg — TensorRT sign-detection node
    │   ├── models/
    │   │   ├── obj.names
    │   │   ├── yolov4-tiny-signs.onnx
    │   │   ├── yolov4-tiny-signs_best_fp16.engine
    │   │   └── yolov4-tiny-signs_fp32_good.engine
    │   └── src/
    │       └── sign_detect_trt.cpp
    └── waypoints_puzzlebot/       # Python pkg — PID controller, state machine, waypoints
        ├── launch/
        │   └── traffic_waypoints.launch.py
        └── waypoints_puzzlebot/
            ├── line_follow.py
            ├── 8_waypoints.py
            ├── pid_identification.py
            └── test_odometry.py
```

---

## System Architecture

```
                          ┌─────────────┐
                          │  cam_publish │  GStreamer / nvarguscamerasrc
                          └──────┬──────┘
                                 │ /cam/img_raw  (sensor_msgs/Image)
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                       ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ line_follower   │   │ traffic_detect   │   │ sign_detect_trt  │
 │  (OpenCV)       │   │  (HSV colour)    │   │  (TensorRT YOLO) │
 └────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
          │                     │                       │
  /line_detector_error   /Traffic_light          /traffic_sign
  /crosswalk_bool                                /stop_area
  /crosswalk_ang
          │                     │                       │
          └─────────────────────▼───────────────────────┘
                         ┌──────────────┐
                         │  line_follow  │  State machine + PID
                         └──────┬───────┘
                                │
                   /VelocitySetL  /VelocitySetR
                                │
                         ┌──────▼──────┐
                         │  Puzzlebot  │
                         └─────────────┘
```

---

## Packages

### vision\_puzzlebot

Pure-Python ROS 2 package (ament\_python). Contains three nodes:

**`cam_publish` (`cam_publish.py`)**
Captures frames from the Jetson CSI camera through a GStreamer/NVMM pipeline and publishes them as `sensor_msgs/Image` on `/cam/img_raw` at a configurable frame rate (default 20 fps, 640×480).

**`line_follower` (`line_follower_camera.py`)**
Processes camera images to produce two outputs:

- **Line following** — crops the bottom quarter of the frame, applies inverted Otsu thresholding and morphological closing, selects the largest contour closest to the image centre, and publishes a signed lateral error in [−1, 1] on `/line_detector_error`.
- **Crosswalk detection** — crops the bottom third of the frame, searches for clusters of six contours with similar area and aspect ratio aligned horizontally (validated with `cv2.fitLine` MSE), publishes the crossing angle on `/crosswalk_ang` and a boolean flag on `/crosswalk_bool`.

**`traffic_detect` (`trafficlight_detect.py`)**
Detects traffic light colour via HSV masking. Configurable HSV ranges for red (two-range wrap-around), yellow, and green. Publishes the detected colour as a `std_msgs/String` on `/Traffic_light`.

---

### vision\_puzzlebot\_trt

C++ ROS 2 package (ament\_cmake). Runs a YOLOv4-tiny model with NVIDIA **TensorRT** for real-time road-sign detection.

- **Engine files** — pre-compiled FP16 and FP32 TensorRT engines included in `models/`. The default is `yolov4-tiny-signs_best_fp16.engine`.
- Subscribes to `/cam/img_raw`, applies a configurable ROI crop, runs NMS post-processing, and publishes the detected sign class on `/traffic_sign` and the bounding-box area on `/stop_area` (used by the controller to estimate distance to a STOP sign).
- Parameters: `confidence_threshold` (default 0.75), `nms_threshold` (default 0.45), `show_window`, ROI extents.

---

### waypoints\_puzzlebot

Pure-Python ROS 2 package (ament\_python). Contains the main controller and auxiliary scripts.

**`line_follow` (`line_follow.py`)**
Centralises all perception inputs and drives the robot through a **state machine** (see below). Uses two independent PID controllers:

- *Angular PID* — tracks the line-follower error (`kp=0.7`, `kd=0.6` by default).
- *Linear PID* — controls advance distance during waypoint manoeuvres (`kp=0.65`).

Odometry is integrated from wheel-encoder velocities (`/VelocityEncL`, `/VelocityEncR`) using a differential-drive kinematic model. A **voting counter** (minimum 3 votes) filters noisy YOLO detections before triggering a state transition.

**`8_waypoints` (`8_waypoints.py`)**
Alternative controller for waypoint-to-waypoint navigation with traffic-light integration, without the full sign-detection pipeline.

**`pid_identification` (`pid_identification.py`)**
Utility node for offline PID parameter identification using wheel-encoder data.

---

## ROS 2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/cam/img_raw` | `sensor_msgs/Image` | pub | Raw camera frames |
| `/line_detector_error` | `std_msgs/Float32` | pub | Signed lateral error [−1, 1] |
| `/crosswalk_bool` | `std_msgs/Bool` | pub | Crosswalk detected (rising edge) |
| `/crosswalk_ang` | `std_msgs/Float32` | pub | Crosswalk alignment angle |
| `/Traffic_light` | `std_msgs/String` | pub | `"red"` / `"yellow"` / `"green"` / `"none"` |
| `/traffic_sign` | `std_msgs/String` | pub | Detected sign class |
| `/stop_area` | `std_msgs/Float32` | pub | STOP sign bounding-box area |
| `/VelocitySetL` | `std_msgs/Float32` | pub | Left wheel velocity setpoint |
| `/VelocitySetR` | `std_msgs/Float32` | pub | Right wheel velocity setpoint |
| `/VelocityEncL` | `std_msgs/Float32` | sub | Left encoder velocity |
| `/VelocityEncR` | `std_msgs/Float32` | sub | Right encoder velocity |

---

## State Machine

The `line_follow` node implements the following states:

```
                  ┌─────────────┐
          ┌──────▶│ LINE_FOLLOW │◀──────────────────────────────────┐
          │       └──────┬──────┘                                   │
          │              │                                           │
          │    crosswalk │             sign detected                 │
          │    detected  │  ┌──────────────────────────────┐        │
          │              ▼  ▼                              │        │
          │       ┌──────────────────┐   stop sign         │  done  │
          │       │  ADVANCE_TO_CROSS│   area threshold     │        │
          │       └────────┬─────────┘   ──────────────────┼──▶ STOP│
          │     aligned    │                                │        │
          └────────────────┘         turn_right sign ──────┼──▶ TURN_RIGHT
                                     turn_left sign  ──────┼──▶ TURN_LEFT
                                     straight sign   ──────┼──▶ STRAIGHT
                                     give_way sign   ──────┼──▶ GIVE_WAY
                                     work_ahead sign ──────┘  (speed flag, no state change)
```

- **`STOP`** — halts the robot until the traffic light turns green.
- **`TURN_RIGHT` / `TURN_LEFT` / `STRAIGHT`** — executes a timed open-loop manoeuvre, then resumes line following.
- **`GIVE_WAY`** — reduces speed momentarily.
- **`ADVANCE_TO_CROSS`** — advances a fixed distance (0.26 m) while correcting angle, then returns to `LINE_FOLLOW`.
- **`work_ahead`** — not a state but a flag that reduces base speed for a configurable duration (2 s, 0.06 m/s).

---

## Detected Signs & Classes

The YOLOv4-tiny model (`obj.names`) recognises six classes:

| Class | Robot action |
|---|---|
| `give_way` | Reduce speed (GIVE\_WAY) |
| `stop` | Stop until green light |
| `straight` | Override intersection, go straight |
| `turn_right` | Execute right turn |
| `work_ahead` | Reduce speed flag (2 s) |
| `turn_left` | Execute left turn |

---

## Configuration

All tunable parameters live in `src/vision_puzzlebot/config/params.yaml` and are loaded at launch time. Key sections:

```yaml
vision_traffic:            # Traffic-light HSV thresholds, minimum contour area
vision_line_follower:      # Camera resolution, FPS, contour area, morph kernel
control_line_follower:     # Wheel geometry, PID gains, speed limits
vision_YOLO:               # TensorRT engine path, confidence threshold
```

To tune HSV ranges without recompiling, edit the `*_hsv_lower` / `*_hsv_upper` arrays and relaunch.

---

## Launch Files

**Full pipeline (line following + sign detection + traffic lights):**

```bash
ros2 launch vision_puzzlebot final_launch.launch.py
```

Starts: `camera`, `traffic_detect`, `line_follower`, `line_follow` (controller), `sign_detect_trt`.

**Waypoint + traffic-light only:**

```bash
ros2 launch waypoints_puzzlebot traffic_waypoints.launch.py
```

Starts: `micro_ros_agent` (serial), `traffic_detect`, `waypoints`.

**Line follower standalone:**

```bash
ros2 launch vision_puzzlebot puzzlebot_line_follower.launch.py
```

---

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| ROS 2 | Humble | ament\_python + ament\_cmake |
| OpenCV | ≥ 4.5 | `python3-opencv` |
| NumPy | ≥ 1.21 | `python3-numpy` |
| cv\_bridge | — | ROS ↔ OpenCV bridge |
| NVIDIA TensorRT | ≥ 8 | For `vision_puzzlebot_trt` |
| CUDA | ≥ 11 | Required by TensorRT |
| rclcpp / rclpy | — | ROS 2 client libraries |
| micro\_ros\_agent | — | For Puzzlebot firmware bridge |

---

## Build & Run

```bash
# Clone or copy the workspace
cd ~/ros2_ws

# Install Python dependencies
pip3 install numpy opencv-python

# Build all packages
colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Launch the full pipeline
ros2 launch vision_puzzlebot final_launch.launch.py
```

> **Note:** The TensorRT engine files (`.engine`) are pre-compiled for a specific Jetson hardware/TensorRT version. If you change hardware or upgrade TensorRT, rebuild the engines from the provided `.onnx` files using `trtexec`:
>
> ```bash
> trtexec --onnx=yolov4-tiny-signs.onnx \
>         --saveEngine=yolov4-tiny-signs_best_fp16.engine \
>         --fp16
> ```