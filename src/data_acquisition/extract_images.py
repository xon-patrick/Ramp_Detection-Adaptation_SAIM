import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
import cv2
import numpy as np
import os

BAG_FILE = "test/rosbag2_2025_12_08-19_24_02/rosbag2_2025_12_08-19_24_02_0.db3"  #nume fisier 
IMAGE_TOPIC = "/camera/image_raw/compressed"
OUTPUT_DIR = "data/raw"
FRAME_INTERVAL = 5.0 

def main():
    rclpy.init()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(BAG_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()

    topic_id = None
    msg_type = None

    for tid, name, ttype in topics:
        if name == IMAGE_TOPIC:
            topic_id = tid
            msg_type = ttype
            break

    if topic_id is None:
        print(f"Topicul {IMAGE_TOPIC} nu a fost gasit")
        return

    msg_class = get_message(msg_type)

    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (topic_id,))
    rows = cursor.fetchall()

    last_saved_time = None

    for timestamp, data in rows:
        time_sec = timestamp / 1e9
        sec = timestamp // 1_000_000_000
        nsec = timestamp % 1_000_000_000

        if last_saved_time is None or (time_sec - last_saved_time) >= FRAME_INTERVAL:

            msg = deserialize_message(data, msg_class)

            if not msg.data or len(msg.data) < 10:
                print(f"Mesaj corupt sau gol la t={time_sec:.2f}s - skipping")
                continue

            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_img is None:
                print(f"Imagine invalida la t={time_sec:.2f}s - skipping")
                continue

            filename = f"{OUTPUT_DIR}/camera_image_raw_compressed-{sec}-{nsec}.png"
            cv2.imwrite(filename, cv_img)

            print(f"Salvat: {filename}")
            last_saved_time = time_sec

    conn.close()
    print("Gata")

if __name__ == "__main__":
    main()