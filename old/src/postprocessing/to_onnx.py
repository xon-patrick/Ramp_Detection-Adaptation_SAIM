from pathlib import Path
from ultralytics import YOLO

model_path = Path(__file__).parent.parent.parent / "trained_models" / "model_005_best.pt"

model = YOLO(str(model_path))

model.export(format="onnx")