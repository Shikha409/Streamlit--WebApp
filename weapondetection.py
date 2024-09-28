import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import time

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# Streamlit app
st.title("YOLOv8 Objects Detection App")

# Sidebar for options
st.sidebar.title("Detection Options Config")
option = st.sidebar.radio("1. Select Input Type:", ("Upload Image", "Upload Video", "Livecam Detection"))

# Sidebar settings
MAX_BOXES_TO_DRAW = st.sidebar.number_input('2. Maximum Boxes To Draw', value=5, min_value=1, max_value=20)
DEVICES = st.sidebar.selectbox("3. Select Device", ['cpu', '0', '1', '2'], index=0)
MIN_SCORE_THRES = st.sidebar.slider('4. Min Confidence Score Threshold', min_value=0.0, max_value=1.0, value=0.4)
save_option = st.sidebar.selectbox("5. Save Result?", ("Yes", "No"))

# Set the model to use the selected device
model.to(DEVICES)

# Function to process image
def process_image(image):
    results = model(image, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)
    return results[0]

# Function to process video frame
def process_video_frame(frame):
    results = model(frame, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)
    return results[0]

# Function to display detection results
def display_results(result, original_image):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Image")
        st.image(original_image, use_column_width=True)
    with col2:
        st.write("Detected Objects")
        st.image(result.plot(), use_column_width=True)
    
    speed_info = result.speed
    st.write(f"Image Details: {original_image.size[1]}x{original_image.size[0]}")
    st.write(f"Objects Detected: {len(result.boxes)}")
    st.write(f"Classes: {', '.join([model.names[int(cls)] for cls in result.boxes.cls])}")
    st.write(f"Speed: {speed_info['preprocess']:.1f}ms preprocess, {speed_info['inference']:.1f}ms inference, {speed_info['postprocess']:.1f}ms postprocess")

# Image upload processing
if option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        result = process_image(image)
        display_results(result, image)
        
        if save_option == "Yes":
            result_image = result.plot()
            st.sidebar.image(result_image, caption="Detected Image", use_column_width=True)
            st.sidebar.download_button("Download Result Image", cv2.imencode('.jpg', result_image)[1].tobytes(), "result_image.jpg")

# Video upload processing
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Create a placeholder for snapshots
        snapshot_placeholder = st.empty()
        
        # Create columns for snapshot controls
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            snapshot_button = st.button("Take Snapshot")
        with col2:
            auto_snapshot = st.checkbox("Auto Snapshot")
        with col3:
            snapshot_interval = st.number_input("Snapshot Interval (seconds)", min_value=1, value=5)
        
        last_snapshot_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = process_video_frame(frame)
            processed_frame = result.plot()
            
            current_time = time.time()
            if snapshot_button or (auto_snapshot and current_time - last_snapshot_time >= snapshot_interval):
                snapshot_placeholder.image(processed_frame, channels="BGR", caption="Latest Snapshot", use_column_width=True)
                last_snapshot_time = current_time
                snapshot_button = False  # Reset the button state
            
            # Display frame info
            st.write(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES):.0f}, "
                     f"Objects: {len(result.boxes)}, "
                     f"Classes: {', '.join([model.names[int(cls)] for cls in result.boxes.cls])}")
        
        cap.release()
        os.unlink(tfile.name)

# Set the model to use the selected device
model.to(DEVICES)

# Webcam Detection Toggle
option = st.selectbox("Select Mode", ["Choose...", "Livecam Detection"])

if option == "Livecam Detection":
    # Start Webcam Detection button
    start_button = st.button("Start Webcam Detection")

    if start_button:
        # After the start button is clicked, show options for webcam or IP camera
        camera_option = st.selectbox("Select Camera Type", ["Webcam", "IP Camera"])

        # If IP Camera is selected, allow user to input the IP stream URL
        if camera_option == "IP Camera":
            ip_url = st.text_input("Enter IP Camera URL (e.g., rtsp:// or http://)", "")
        else:
            ip_url = None

        run = True

        if run:
            # If using IP Camera, try to connect to the stream
            if ip_url:
                camera = cv2.VideoCapture(ip_url)
                if not camera.isOpened():
                    st.error(f"Failed to connect to IP Camera at {ip_url}. Please check the URL and try again.")
                    run = False
                else:
                    st.success(f"Successfully connected to IP Camera: {ip_url}")
            else:
                # Try different camera indices for the Webcam
                for camera_index in [0, 1, 2, 3]:
                    camera = cv2.VideoCapture(camera_index)
                    if camera.isOpened():
                        st.success(f"Successfully connected to Webcam index {camera_index}")
                        break
                else:
                    st.error("Failed to connect to any Webcam. Please check your camera connection and try again.")
                    run = False

            # Display video feed and perform detection
            if run:
                FRAME_WINDOW = st.image([])
                info_text = st.empty()

                try:
                    while run:
                        ret, frame = camera.read()
                        if not ret:
                            st.warning("Failed to capture frame. Trying again...")
                            time.sleep(1)
                            continue

                        result = model(frame, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)

                        # Change detection box color (RGB value) and line thickness
                        processed_frame = result[0].plot(line_width=3, color=(0, 255, 0))  # Green boxes with thicker lines

                        # Resize the frame to a more viewable size
                        resized_frame = cv2.resize(processed_frame, (640, 480))

                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                        # Display the frame
                        FRAME_WINDOW.image(rgb_frame, caption="Live Camera Feed")

                        # Display detected objects
                        detected_objects = [
                            f"{model.names[int(cls)]} ({conf:.2f})"
                            for cls, conf in zip(result[0].boxes.cls, result[0].boxes.conf)
                        ]
                        info_text.write(f"Detected Objects: {', '.join(detected_objects)}")

                        # Add a small delay to control frame rate
                        time.sleep(0.1)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    camera.release()
else:
    st.write("Click 'Livecam Detection' and then 'Start Webcam Detection' to begin.")

st.write("Note: Press 'Stop' in the top right corner or uncheck the 'Start Camera Detection' box to end the detection.")
