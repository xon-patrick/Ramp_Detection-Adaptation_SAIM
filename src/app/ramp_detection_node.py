#!/usr/bin/env python3
"""
Simple ROS2 ramp detector (real-time).
- Subscribes: /camera/image_raw/compressed
- Publishes: /ramp/detections_image, /ramp/markers, /ramp/robot_state
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from geometry_msgs.msg import Point, PointStamped
import tf2_ros
from cv_bridge import CvBridge


CLASS_COLORS = {
    "rampDown": (255, 0, 0),
    "rampUp": (0, 165, 255),
    "ramps-railing": (0, 255, 0),
}


class RampDetectionNode(Node):
    def __init__(self):
        super().__init__("ramp_detection_node")

        self.declare_parameter("model_path", "models/trained_model_v1.onnx")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("marker_distance", 2.0)
        self.declare_parameter("path_min_step", 0.3)

        self.conf_threshold = float(self.get_parameter("confidence_threshold").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.marker_distance = float(self.get_parameter("marker_distance").value)
        self.path_min_step = float(self.get_parameter("path_min_step").value)

        self.bridge = CvBridge()
        self.camera_info = None
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0

        model_path = self._resolve_model_path(self.get_parameter("model_path").value)
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_image = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, qos_image
        )
        self.sub_cam = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.camera_info_callback, 10
        )

        self.pub_image = self.create_publisher(Image, "/ramp/detections_image", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/ramp/markers", 10)
        self.pub_state = self.create_publisher(String, "/ramp/robot_state", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.path_up = []
        self.path_down = []
        
        self.last_railing_time = 0
        self.railing_timeout = 5.0
        self.status_text = "IDLE"

        self.get_logger().info("Ramp detection node ready")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Confidence: {self.conf_threshold}")
        self.get_logger().info(f"Map frame: {self.map_frame}")
        self.get_logger().info(f"Base frame: {self.base_frame}")

    def _resolve_model_path(self, rel_path: str) -> str:
        project_root = Path(__file__).resolve().parents[2]
        model_path = (project_root / rel_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return str(model_path)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def run_inference(self, image: np.ndarray):
        detections = []
        results = self.model(image, imgsz=640, conf=self.conf_threshold, verbose=False)
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_name = self.class_names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                }
            )

        return detections

    def _point_in_map(self, header, cx, cy):
        point = PointStamped()
        point.header = header
        point.header.frame_id = self.base_frame

        angle_x = (cx - self.cx) / self.fx
        angle_y = (cy - self.cy) / self.fy
        point.point.x = self.marker_distance
        point.point.y = self.marker_distance * angle_x
        point.point.z = self.marker_distance * angle_y

        try:
            return self.tf_buffer.transform(point, self.map_frame)
        except Exception:
            return point

    def _maybe_add_path_point(self, path, point):
        if not path:
            path.append(point)
            return
        last = path[-1]
        dx = point.x - last.x
        dy = point.y - last.y
        dz = point.z - last.z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist >= self.path_min_step:
            path.append(point)

    def image_callback(self, msg: CompressedImage):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Image decode failed: {exc}")
            return

        try:
            detections = self.run_inference(image)
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            return

        Height, Width = image.shape[:2]
        current_time = time.time()
        
        # Check for railings in current frame
        has_railing_now = any(d["cls_name"] == "ramps-railing" for d in detections)
        if has_railing_now:
            self.last_railing_time = current_time
        
        # Time since last railing
        time_since_railing = current_time - self.last_railing_time
        railing_recent = time_since_railing < self.railing_timeout
        
        # Find best ramp
        ramps = [d for d in detections if d["cls_name"] in ["rampUp", "rampDown"]]
        state = "IDLE"
        self.status_text = "IDLE\n(waiting for ramps)"
        
        if ramps:
            best_ramp = max(ramps, key=lambda d: d["conf"])
            ramp_cls = best_ramp["cls_name"]
            x1, y1, x2, y2 = best_ramp["bbox"]
            cx, cy = best_ramp["center"]
            
            # Calculate metrics
            ramp_width = x2 - x1
            ramp_height = y2 - y1
            ramp_area_ratio = (ramp_width * ramp_height) / (Width * Height)
            
            center_x = Width / 2
            is_centered = abs(cx - center_x) < Width * 0.15
            in_lower_region = cy > Height * 0.6
            takes_large_space = ramp_area_ratio > 0.15
            
            direction = "UP" if ramp_cls == "rampUp" else "DOWN"
            
            # Build status
            if takes_large_space and in_lower_region and railing_recent:
                state = f"ON_RAMP_{direction}"
                self.status_text = f"ON RAMP ({direction})\nRailing seen {time_since_railing:.1f}s ago"
            elif not is_centered and in_lower_region and has_railing_now:
                state = f"APPROACHING_RAMP_{direction}"
                self.status_text = f"APPROACHING RAMP ({direction})\nRailing detected now"
            elif detections:
                state = f"RAMP_DETECTED_{direction}"
                self.status_text = f"⚠ RAMP DETECTED ({direction})\n"
                if not is_centered:
                    self.status_text += "• Not centered\n"
                if not in_lower_region:
                    self.status_text += "• Not in lower region\n"
                if not has_railing_now:
                    if railing_recent:
                        self.status_text += f"• Railing seen {time_since_railing:.1f}s ago\n"
                    else:
                        self.status_text += "• No recent railing\n"

        self.pub_state.publish(String(data=state))

        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = msg.header
            self.pub_image.publish(img_msg)
        except Exception as exc:
            self.get_logger().error(f"Image publish failed: {exc}")

        marker_array = MarkerArray()
        for i, det in enumerate(detections):
            point_map = self._point_in_map(msg.header, det["center"][0], det["center"][1])

            marker = Marker()
            marker.header = point_map.header
            marker.ns = "ramps"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = point_map.point.x
            marker.pose.position.y = point_map.point.y
            marker.pose.position.z = point_map.point.z

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            color = CLASS_COLORS.get(det["cls_name"], (255, 255, 255))
            marker.color.r = color[2] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[0] / 255.0
            marker.color.a = 0.8
            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

            if point_map.header.frame_id == self.map_frame:
                if det["cls_name"] == "rampUp":
                    self._maybe_add_path_point(self.path_up, marker.pose.position)
                elif det["cls_name"] == "rampDown":
                    self._maybe_add_path_point(self.path_down, marker.pose.position)

        if self.path_up:
            up_marker = Marker()
            up_marker.header.frame_id = self.map_frame
            up_marker.header.stamp = msg.header.stamp
            up_marker.ns = "ramp_path_up"
            up_marker.id = 1000
            up_marker.type = Marker.LINE_STRIP
            up_marker.action = Marker.ADD
            up_marker.scale.x = 0.05
            up_marker.color.r = 0.0
            up_marker.color.g = 0.6
            up_marker.color.b = 1.0
            up_marker.color.a = 0.9
            up_marker.points = list(self.path_up)
            marker_array.markers.append(up_marker)

        if self.path_down:
            down_marker = Marker()
            down_marker.header.frame_id = self.map_frame
            down_marker.header.stamp = msg.header.stamp
            down_marker.ns = "ramp_path_down"
            down_marker.id = 1001
            down_marker.type = Marker.LINE_STRIP
            down_marker.action = Marker.ADD
            down_marker.scale.x = 0.05
            down_marker.color.r = 1.0
            down_marker.color.g = 0.2
            down_marker.color.b = 0.2
            down_marker.color.a = 0.9
            down_marker.points = list(self.path_down)
            marker_array.markers.append(down_marker)

        # Add status text marker
        status_marker = Marker()
        status_marker.header.frame_id = self.map_frame
        status_marker.header.stamp = msg.header.stamp
        status_marker.ns = "status"
        status_marker.id = 2000
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.text = self.status_text
        status_marker.scale.z = 0.5
        status_marker.color.r = 1.0
        status_marker.color.g = 1.0
        status_marker.color.b = 1.0
        status_marker.color.a = 1.0
        status_marker.pose.position.x = 0.0
        status_marker.pose.position.y = 0.0
        status_marker.pose.position.z = 2.0
        status_marker.lifetime.sec = 1
        marker_array.markers.append(status_marker)

        self.pub_markers.publish(marker_array)


def main():
    rclpy.init()
    node = RampDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
