
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import subprocess
import imageio_ffmpeg
import os

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")

st.title("🪖 Helmet Detection System")
st.write("Upload an image or video to detect helmets and persons.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.write("Model Classes:", model.names)

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    if uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect"):

            image_array = np.array(image)

            results = model.predict(
                image_array,
                conf=0.25,
                imgsz=640,
                verbose=False
            )

            output = results[0].plot()
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            st.subheader("Detection Results")
            st.image(output, use_container_width=True)

    else:

        st.video(uploaded_file)

        if st.button("Detect Video"):

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as f:
                f.write(uploaded_file.getbuffer())
                input_path = f.name

            temp_output = input_path.replace(".mp4", "_temp.mp4")
            output_path = input_path.replace(".mp4", "_detected.mp4")

            cap = cv2.VideoCapture(input_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                temp_output,
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

                results = model.predict(
                    frame,
                    conf=0.25,
                    imgsz=640,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                count += 1

                if total > 0:
                    progress.progress(min(count / total, 1.0))

            cap.release()
            writer.release()

            if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
                st.error("Video processing failed!")
                st.stop()

            try:

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i", temp_output,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        output_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

                st.success("Video detection completed!")

                st.subheader("Detection Results")
                st.video(output_path)

                with open(output_path, "rb") as video:
                    st.download_button(
                        "Download Detected Video",
                        data=video,
                        file_name="helmet_detected.mp4",
                        mime="video/mp4"
                    )

            except subprocess.CalledProcessError as e:
                st.error("Video conversion failed!")
                st.code(e.stderr[-3000:])

            except Exception as e:
                st.error("Error processing video!")
                st.exception(e)

            finally:
                for path in [input_path, temp_output]:
                    if os.path.exists(path):
                        os.remove(path)
