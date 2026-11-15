# 🪖 Helmet Detection System

## 📌 Project Overview
The Helmet Detection System is a Deep Learning project that uses YOLO11n to detect helmets and persons in images and videos.

The application is developed using Streamlit, allowing users to upload images or videos and view detection results with bounding boxes.

## ✨ Features
- Detect helmets and persons in images.
- Detect helmets and persons in videos.
- Display bounding boxes around detected objects.
- Preserve original image and video colors.
- Simple and user-friendly Streamlit interface.
- Real-time detection processing for uploaded media.

## 🛠️ Technologies Used
- Python
- YOLO11n
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pillow

## 📂 Project Structure

```text
image_detection_project/
│
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project folder**

```bash
cd image_detection_project
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

## 📸 How to Use
1. Open the Streamlit application.
2. Upload an image or video.
3. Click the Detect or Detect Video button.
4. View the detection results with bounding boxes.

## 🤖 Model Information
- Model: YOLO11n
- Task: Object Detection
- Classes: Helmet, Person
- Dataset: Worker Helmet Detection Dataset



