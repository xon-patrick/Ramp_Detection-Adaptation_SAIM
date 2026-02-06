from ultralytics import YOLO
from PIL import Image

model = YOLO("runs\\detect\\train\\weights\\best.pt")

image_path = "D:\\An3S1\\RN\\Proiect Rn\\Ramp_Detection-Adaptation_SAIM\\data\\test\\images\\camera_image_raw_compressed-1765214732-232190833_png.rf.799b4bb78b22bafa565f803b112771c1.jpg"
image = Image.open(image_path)

results = model(image_path, imgsz=640)

annotated = results[0].plot()
annotated_image = Image.fromarray(annotated)
annotated_image.show()
