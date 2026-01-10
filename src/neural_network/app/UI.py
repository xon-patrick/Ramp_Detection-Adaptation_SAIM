# pentru rulare "streamlit run src/neural_network/app/UI.py"
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path

st.set_page_config(page_title="Ramp Detection", layout="wide")
st.title("SAIM Ramp Detection – YOLOv8")


# @st.cache_resource
# def load_model():
#     model_path = Path(__file__).resolve().parent.parent / "models" / "best.pt"
#     return YOLO(str(model_path))

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "runs" / "train" / "exp_20260110_125217" / "weights" / "best.pt"
    return YOLO(str(model_path))


model = load_model()

uploaded_file = st.file_uploader("Incarca o imagine", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Imagine originala", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        results = model(tmp.name, imgsz=640, conf=0.2)

    annotated = results[0].plot()
    st.image(annotated, caption="Detectii", use_container_width=True)

    if len(results[0].boxes) > 0:
        st.write("Obiecte detectate:")
        for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
            st.write(f"Clasa: {model.names[int(cls)]}, Confidence: {conf:.2f}")
    else:
        st.write("Nu s-au detectat obiecte.")
