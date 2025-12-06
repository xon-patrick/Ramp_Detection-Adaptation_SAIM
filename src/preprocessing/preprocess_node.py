import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import json
import cv2
import os
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

class PreprocessNode(Node):
    def __init__(self):
        super().__init__('preprocess_node')

        self.bridge = CvBridge()

        # topicuri
        self.image_sub = Subscriber(self, Image, '/camera/compressed_raw')
        self.lidar_sub = Subscriber(self, LaserScan, '/scan')

        # imbinare image + lidar
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.save_dir = 'dataset_raw'
        os.makedirs(self.save_dir, exist_ok=True)

        self.counter = 0
        self.get_logger().info("Preprocess node started")

    def callback(self, img_msg, lidar_msg):

        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # disntanta fata de rampa
        ranges = np.array(lidar_msg.ranges)
        forward_window = ranges[0:10].tolist() + ranges[-10:] 
        forward_dist = float(np.nanmin(forward_window))


        img_path = os.path.join(self.save_dir, f"frame_{self.counter:06d}.jpg")
        cv2.imwrite(img_path, frame)

        # metadata
        meta = {
            "frame": self.counter,
            "forward_distance": forward_dist,
            "stamp_sec": img_msg.header.stamp.sec,
            "stamp_nanosec": img_msg.header.stamp.nanosec
        }

        meta_path = os.path.join(self.save_dir, f"frame_{self.counter:06d}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

        self.get_logger().info(f"Saved {img_path}")

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
