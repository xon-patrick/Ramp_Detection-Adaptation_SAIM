# Ramp Detection (ROS2)

Minimal node for real-time ramp detection using an ONNX YOLO model.

## Install

```bash
# Use a venv if possible
python3 -m venv ramp_env
source ramp_env/bin/activate
pip install opencv-python onnxruntime 'numpy<2'

# ROS dependency
sudo apt-get install ros-jazzy-cv-bridge
```

If you already have NumPy 2.x and get cv_bridge errors:
```bash
pip install --upgrade --force-reinstall 'numpy<2'
```

## Run

```bash
python3 src/app/ramp_detection_node.py
```

## Topics

Input:
- /camera/image_raw/compressed

Output:
- /ramp/detections_image
- /ramp/markers
- /ramp/robot_state
