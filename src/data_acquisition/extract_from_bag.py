import argparse
import os
import sys
import time
from datetime import datetime

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except Exception:
    print("rclpy and sensor_msgs are required (source your ROS2 environment).", file=sys.stderr)
    raise

try:
    from cv_bridge import CvBridge
except Exception:
    print("cv_bridge is required (install ros-<distro>-cv-bridge).", file=sys.stderr)
    raise

import cv2


class ImageSaver(Node):
    def __init__(self, topic: str, out_dir: str, prefix: str = "img", fmt: str = "png", max_images: int = 0, interval: float = 0.0):
        super().__init__("image_saver")
        self.topic = topic
        self.out_dir = out_dir
        self.prefix = prefix
        self.fmt = fmt.lstrip('.')
        self.max_images = int(max_images)
        self.interval = float(interval)
        self._last_saved_ts = None

        os.makedirs(self.out_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.count = 0
        self.sub = self.create_subscription(Image, self.topic, self.callback, 10)
        self.get_logger().info(f"Subscribed to {self.topic}, saving to {self.out_dir}")

    def callback(self, msg: Image):
        if self.max_images and self.count >= self.max_images:
            return

        # Determine current message timestamp (prefer header stamp)
        try:
            current_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            current_ts = time.time()

        # Skip saving if within the interval
        if self.interval and self._last_saved_ts is not None:
            if (current_ts - self._last_saved_ts) < self.interval:
                return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg)
            except Exception as e:
                self.get_logger().error(f"Failed to convert Image message: {e}")
                return

        # Use header timestamp if available, otherwise use current time
        try:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            timestamp = datetime.fromtimestamp(ts)
            ts_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')
        except Exception:
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        filename = f"{self.prefix}_{self.count:06d}_{ts_str}.{self.fmt}"
        path = os.path.join(self.out_dir, filename)

        try:
            ok = cv2.imwrite(path, cv_image)
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
            self.count += 1
            # record saved timestamp for interval gating
            try:
                self._last_saved_ts = current_ts
            except Exception:
                self._last_saved_ts = time.time()
            if self.count % 50 == 0:
                self.get_logger().info(f"Saved {self.count} images (last: {filename})")
        except Exception as e:
            self.get_logger().error(f"Failed to write image to {path}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Subscribe to a ROS2 Image topic and save images to disk.")
    parser.add_argument("--topic", default="/camera/image_raw", help="ROS2 Image topic to subscribe to")
    parser.add_argument("--output", default="data/raw/new", help="Output directory to save images")
    parser.add_argument("--prefix", default="img", help="Filename prefix")
    parser.add_argument("--format", default="png", help="Image format (png, jpg, etc.)")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of images to save (0 = unlimited)")
    parser.add_argument("--interval", type=float, default=0.0, help="Minimum seconds between saved frames (0 = every frame)")

    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.output)

    rclpy.init()
    node = ImageSaver(topic=args.topic, out_dir=out_dir, prefix=args.prefix, fmt=args.format, max_images=args.max, interval=args.interval)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
