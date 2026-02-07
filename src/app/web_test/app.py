import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path
import time

st.set_page_config(page_title="Ramp Detection", layout="wide")
st.title("SAIM Ramp Detection – YOLOv8")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[3] / "models" / "trained_model_v1.onnx"
    return YOLO(str(model_path))


def check_ramp_railing_proximity(boxes, names):
    ramp_boxes = []
    railing_boxes = []

    for i, cls in enumerate(boxes.cls):
        class_name = names[int(cls)]
        box = boxes.xyxy[i].cpu().numpy()

        if "ramp" in class_name.lower() and "railing" not in class_name.lower():
            ramp_boxes.append(box)
        elif "railing" in class_name.lower():
            railing_boxes.append(box)

    # check proximity
    for ramp_box in ramp_boxes:
        for railing_box in railing_boxes:
            x_overlap = ramp_box[0] < railing_box[2] and ramp_box[2] > railing_box[0]
            y_overlap = ramp_box[1] < railing_box[3] and ramp_box[3] > railing_box[1]

            if x_overlap and y_overlap:
                return True

    return False


model = load_model()

# init state
if 'following_ramp' not in st.session_state:
    st.session_state.following_ramp = False
if 'follow_start_time' not in st.session_state:
    st.session_state.follow_start_time = None
if 'path_finished' not in st.session_state:
    st.session_state.path_finished = False
if 'last_uploaded_name' not in st.session_state:
    st.session_state.last_uploaded_name = None

# separare pe coloane
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Imagini")
    uploaded_file = st.file_uploader("Incarca o imagine", type=["jpg", "png", "jpeg"])

# reset on new img
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.following_ramp = False
    st.session_state.follow_start_time = None
    st.session_state.path_finished = False
    st.session_state.last_uploaded_name = uploaded_file.name

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    with col1:
        st.image(image, caption="Imagine originala", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        results = model(tmp.name, imgsz=640, conf=0.2)

    annotated = results[0].plot()

    with col1:
        st.image(annotated, caption="Detectii", use_container_width=True)

    with col2:
        st.subheader("Informatii Detectie")

        if len(results[0].boxes) > 0:
            st.write("**Obiecte detectate:**")
            for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
                st.write(f"• **{model.names[int(cls)]}** - Confidence: {conf:.2f}")

            ramp_railing_close = check_ramp_railing_proximity(results[0].boxes, model.names)

            st.divider()

            if st.session_state.following_ramp and st.session_state.follow_start_time:
                elapsed = time.time() - st.session_state.follow_start_time
                if elapsed < 20:
                    remaining = 20 - int(elapsed)
                    st.success(f"**Following ramp...** ({remaining}s remaining)")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.session_state.following_ramp = False
                    st.session_state.path_finished = True
                    st.session_state.follow_start_time = None

            if st.session_state.path_finished:
                st.info(" **Path finished**")
            elif not st.session_state.following_ramp:
                if ramp_railing_close:
                    st.success("Ramp and railing detected in proximity!")
                    if st.button("-> Follow Ramp Path", type="primary", use_container_width=True):
                        st.session_state.following_ramp = True
                        st.session_state.follow_start_time = time.time()
                        st.session_state.path_finished = False
                        st.rerun()
                else:
                    st.warning("Unsure of ramp path")
                    st.button("-> Follow Ramp Path", type="secondary", disabled=True, use_container_width=True)
        else:
            st.write("Nu s-au detectat obiecte.")
else:
    with col2:
        st.info("Incarca o imagine pentru a incepe detectia")
