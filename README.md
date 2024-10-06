# Streamlit--WebApp
# YOLOv8 Object Detection App
This Streamlit application leverages the YOLOv8 model to perform real-time object detection on images, videos, and live webcam feeds. The app is designed with an easy-to-use interface and includes a variety of options to customize detection settings.

# Features
#Image, Video, and Live Webcam Detection:

Upload images or videos for object detection.
Option for live webcam detection, with support for both local webcams and IP cameras.
#Object Detection using YOLOv8:

Utilizes YOLOv8 for object detection with options for CPU or GPU processing.
Adjustable detection settings like the maximum number of objects to detect and minimum confidence threshold.
#Detection Results Display:

Side-by-side comparison of original and processed (detected objects) images or frames.
Provides detection details including object classes, confidence scores, and speed metrics (preprocess, inference, and postprocess times).
#Snapshot and Video Saving:

In video mode, take snapshots or set up auto-snapshots at specified intervals.
Save processed images and videos with detected objects highlighted.
#Real-Time Performance Metrics:

Displays frame processing details such as detection counts, detected classes, and performance speeds for each frame in live detection.
# How It Works
1. Upload Image
You can upload images in formats such as .jpg, .jpeg, .png, .bmp, and .webp.
The app processes the image using YOLOv8 and shows the detected objects along with their classes and confidence scores.
Option to save the processed image with bounding boxes.

2. Upload Video
Upload a video file (.mp4, .avi, .mov).
The app processes each frame of the video and displays detection results in real-time.
Provides a snapshot feature to capture frames with detected objects.
Option to save the processed video with bounding boxes.

3. Livecam Detection
Detect objects in real-time using a live webcam or IP camera feed.
Displays live frames with detected objects and performance details.
Provides a detection history for each processed frame, including object counts and processing speed.
Supports both webcam detection and IP camera streams (RTSP/HTTP).

# Sidebar Configurations
**Input Type**: Choose between Upload Image, Upload Video, or Livecam Detection.
**Maximum Boxes to Draw:** Specify the maximum number of objects to detect per image/frame.
**Device Selection:** Choose whether to run on CPU or available GPUs (if configured).
**Minimum Confidence Score:** Set the threshold for object detection confidence.
**Save Option:** Select whether to save the detection results (images or videos).
# Setup and Installation
# Clone the Repository:


!git clone https://github.com/your-repository-name/YOLOv8-Object-Detection-App.git
cd YOLOv8-Object-Detection-App
 **Install Dependencies:** You can install the required dependencies using pip:


pip install -r requirements.txt
**Download YOLOv8 Weights:**

Download your YOLOv8 pre-trained weights (best.pt) and place them in the weights/ directory.
**Run the App:** Run the Streamlit app with the following command:


streamlit run app.py
**Access the App:** Open your browser and go to http://localhost:8501 to access the object detection app.

**Dependencies
Python 3.8+
Streamlit
OpenCV
Pillow
ultralytics (YOLO)**
To install the required Python packages, run:


! pip install streamlit opencv-python-headless Pillow ultralytics
**Future Improvements**
Adding support for multiple YOLO models (YOLOv5, YOLOv6, etc.).
Implementing custom object detection classes.
Integration with cloud services for real-time monitoring.
**License**
This project is licensed under the MIT License. See the LICENSE file for details.

**stremlit web app link :** https://yolov8-detection--webapp-d9evvvch8gpkbtzhserfbb.streamlit.app/
