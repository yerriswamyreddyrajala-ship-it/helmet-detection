
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")

st.title("🪖 Helmet Detection System")
st.write("Upload an image or video to detect helmets and persons.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect"):
            image_rgb = np.array(image)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            results = model.predict(image_bgr, verbose=False)
            output = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            st.subheader("Detection Results")
            st.image(output, use_container_width=True)

    else:
        st.video(uploaded_file)

        if st.button("Detect Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded_file.getbuffer())
                input_path = f.name

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                output_path = f.name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, verbose=False)
                writer.write(results[0].plot())

                count += 1
                if total:
                    progress.progress(min(count / total, 1.0))

            cap.release()
            writer.release()

            st.success("Video detection completed!")

            with open(output_path, "rb") as video:
                st.video(video.read())
