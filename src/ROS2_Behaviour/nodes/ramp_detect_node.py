import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

CLASS_NAMES = ['ramp_up', 'ramp_down', 'ramp_railing']

class RampDetector(Node):
    def __init__(self):
        super().__init__('ramp_detector_node')

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw/image',
            self.image_cb,
            10
        )

        self.pub = self.create_publisher(
            Detection2DArray,
            '/ramp/detections',
            10
        )

        self.bridge = CvBridge()
        self.model = YOLO('model.onnx')
        self.conf_thres = 0.5

        self.get_logger().info('YOLOv8 Ramp detector ready')

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(img, conf=self.conf_thres, verbose=False)

        det_array = Detection2DArray()
        det_array.header = msg.header

        if len(results) == 0:
            return

        boxes = results[0].boxes
        if boxes is None:
            self.pub.publish(det_array)
            return

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det = Detection2D()
            det.bbox.center.x = (x1 + x2) / 2.0
            det.bbox.center.y = (y1 + y2) / 2.0
            det.bbox.size_x = (x2 - x1)
            det.bbox.size_y = (y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.id = CLASS_NAMES[cls_id]
            hyp.score = conf

            det.results.append(hyp)
            det_array.detections.append(det)

        self.pub.publish(det_array)

def main():
    rclpy.init()
    node = RampDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
