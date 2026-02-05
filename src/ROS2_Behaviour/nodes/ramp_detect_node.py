import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
from math import atan2, sqrt

CLASS_NAMES = ['ramp_up', 'ramp_down', 'ramp_railing']

class RampDetector(Node):
    def __init__(self):
        super().__init__('ramp_detector_node')

        self.sub_image = self.create_subscription(
            Image,
            '/camera/image_raw/compressed',
            self.image_cb,
            10
        )

        self.sub_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_cb,
            10
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray,
            '/ramp/detections',
            10
        )

        self.pub_markers = self.create_publisher(
            MarkerArray,
            '/ramp/markers',
            10
        )

        self.pub_image = self.create_publisher(
            Image,
            '/ramp/detections_image',
            10
        )

        self.bridge = CvBridge()
        self.model = YOLO('model_005_best.onnx')
        self.conf_thres = 0.5

        # Camera intrinsics (adjust these for your camera)
        self.focal_length = 500.0  # pixels
        self.image_center_x = 315.0  # half of image width (630/2)
        self.image_center_y = 315.0  # half of image height (630/2)
        
        # Lidar data
        self.lidar_ranges = None
        self.lidar_angles = None
        self.lidar_frame = None

        self.get_logger().info('YOLOv8 Ramp detector ready')

    def lidar_cb(self, msg):
        """Callback for lidar scan data"""
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_frame = msg.header.frame_id
        
        # Calculate angles for each beam
        num_readings = len(msg.ranges)
        self.lidar_angles = np.linspace(
            msg.angle_min, 
            msg.angle_max, 
            num_readings
        )

    def bbox_collision(self, box1, box2):
        """Check if two bounding boxes collide (x1,y1,x2,y2 format)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # AABB collision detection
        if x1_max < x2_min or x2_max < x1_min:
            return False
        if y1_max < y2_min or y2_max < y1_min:
            return False
        return True

    def get_range_from_lidar(self, x_pixel, y_pixel, img_width=630, img_height=630):
        """
        Estimate distance to detection based on lidar scan.
        Maps image coordinates to lidar beam angle.
        """
        if self.lidar_ranges is None or self.lidar_angles is None:
            return None
        
        # Map image x-coordinate to lidar angle (simple linear mapping)
        # Assumes lidar field of view matches camera field of view
        lidar_fov = self.lidar_angles[-1] - self.lidar_angles[0]
        camera_fov_horizontal = 2 * np.arctan(img_width / (2 * self.focal_length))
        
        # Map x pixel to angle
        angle_offset = (x_pixel - self.image_center_x) / img_width * camera_fov_horizontal
        angle_offset = np.clip(angle_offset, self.lidar_angles[0], self.lidar_angles[-1])
        
        # Find closest lidar beam
        beam_idx = np.argmin(np.abs(self.lidar_angles - angle_offset))
        
        # Get range from that beam
        lidar_range = self.lidar_ranges[beam_idx]
        
        # Filter invalid readings
        if not np.isfinite(lidar_range) or lidar_range <= 0.0:
            return None
        
        return lidar_range

    def create_marker(self, det_id, class_name, bbox_center, range_dist, header):
        """Create a visualization marker for RViz"""
        marker = Marker()
        marker.header = header
        marker.id = det_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set position based on range and image coordinate
        x_pixel = bbox_center[0]
        img_width = 640
        
        # Calculate angle to detection
        camera_fov_horizontal = 2 * np.arctan(img_width / (2 * self.focal_length))
        angle_to_det = (x_pixel - self.image_center_x) / img_width * camera_fov_horizontal
        
        if range_dist is not None:
            marker.pose.position.x = range_dist * np.cos(angle_to_det)
            marker.pose.position.y = range_dist * np.sin(angle_to_det)
            marker.pose.position.z = 0.0
        else:
            return None
        
        # Size
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Color based on class
        if 'up' in class_name.lower():
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.65, 0.0  # Orange
        elif 'down' in class_name.lower():
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0  # Blue
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0  # Yellow
        marker.color.a = 0.8
        
        return marker

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        try:
            results = self.model(img, conf=self.conf_thres, verbose=False)
        except Exception as e:
            self.get_logger().error(f'Model inference failed: {e}')
            return

        det_array = Detection2DArray()
        det_array.header = msg.header
        marker_array = MarkerArray()

        if len(results) == 0:
            self.pub_detections.publish(det_array)
            self.pub_markers.publish(marker_array)
            return

        boxes = results[0].boxes
        if boxes is None:
            self.pub_detections.publish(det_array)
            self.pub_markers.publish(marker_array)
            return

        # Separate detections by class
        ramp_ups = []
        ramp_downs = []
        railings = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            det_info = {
                'cls_id': cls_id,
                'cls_name': CLASS_NAMES[cls_id],
                'conf': conf,
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            }
            
            if cls_id == 0:  # ramp_up
                ramp_ups.append(det_info)
            elif cls_id == 1:  # ramp_down
                ramp_downs.append(det_info)
            elif cls_id == 2:  # ramp_railing
                railings.append(det_info)

        # Process ramps: only keep if colliding with railing
        valid_ramps = []
        
        for ramp in ramp_ups + ramp_downs:
            for railing in railings:
                if self.bbox_collision(ramp['bbox'], railing['bbox']):
                    valid_ramps.append(ramp)
                    break  # Found a colliding railing, ramp is valid

        # Publish detections and markers
        marker_id = 0
        for ramp in valid_ramps:
            x1, y1, x2, y2 = ramp['bbox']
            center_x, center_y = ramp['center']
            
            # Get range from lidar
            range_dist = self.get_range_from_lidar(center_x, center_y)
            
            # Create detection message
            det = Detection2D()
            det.bbox.center.x = center_x
            det.bbox.center.y = center_y
            det.bbox.size_x = (x2 - x1)
            det.bbox.size_y = (y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.id = ramp['cls_name']
            hyp.score = ramp['conf']

            det.results.append(hyp)
            det_array.detections.append(det)

            # Create marker for RViz
            if range_dist is not None:
                marker = self.create_marker(
                    marker_id, 
                    ramp['cls_name'], 
                    ramp['center'], 
                    range_dist, 
                    msg.header
                )
                if marker:
                    marker_array.markers.append(marker)
                    marker_id += 1

        # Publish
        self.pub_detections.publish(det_array)
        self.pub_markers.publish(marker_array)
        
        # Publish annotated image
        annotated_img = img.copy()
        for ramp in valid_ramps:
            x1, y1, x2, y2 = ramp['bbox']
            color = (0, 165, 255) if 'up' in ramp['cls_name'].lower() else (255, 0, 0)  # BGR: orange for up, blue for down
            cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_img, f"{ramp['cls_name']} {ramp['conf']:.2f}", 
                       (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        annotated_msg.header = msg.header
        self.pub_image.publish(annotated_msg)

def main():
    rclpy.init()
    node = RampDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
