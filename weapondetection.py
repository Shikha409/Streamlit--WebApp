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
enable_cpu = st.sidebar.checkbox("Enable CPU Processing", value=True)
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

    st.subheader("Detection Frame Detail Results")
    
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
            st.sidebar.title("Detected Image Result")
            st.sidebar.image(result_image, caption="Detected Image", use_column_width=True)
            st.sidebar.download_button("Download Result Image", cv2.imencode('.jpg', result_image)[1].tobytes(), "result_image.jpg")

# Function to load YOLOR and process each frame with DeepSORT
def load_yolor_and_process_each_frame(frame, min_score_thresh,enable_cpu ):
     results = model(frame, conf=min_score_thresh, max_det=MAX_BOXES_TO_DRAW)
     return results

# Video upload processing
if option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.subheader("Upload ForTested Video ")
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        
        
        st.video(tfile.name)

        st.subheader("Detection Output Video ")

        
        cap = cv2.VideoCapture(tfile.name)
        
        # Create a placeholder for snapshots
        snapshot_placeholder = st.empty()
        # Create a placeholder for video processing
        video_placeholder = st.empty()
        
        # Create columns for snapshot controls
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            snapshot_button = st.button("Take Snapshot")
        with col2:
            auto_snapshot = st.checkbox("Auto Snapshot")
        with col3:
            snapshot_interval = st.number_input("Snapshot Interval (seconds)", min_value=1, value=5)
        
        last_snapshot_time = time.time()

        st.subheader("Detection Frame Detail Results")
        
        processed_frames = []
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


             # Call the function to process each frame with YOLOR and DeepSORT
            results = load_yolor_and_process_each_frame(frame, MIN_SCORE_THRES, enable_cpu)
            
             # Draw bounding boxes on the frame
            processed_frame = results[0].plot()  # Assuming you want to use YOLO's plot method

            # Extract speed information
            speed_info = results[0].speed if isinstance(results[0].speed, dict) else {'inference': 0, 'preprocess': 0, 'postprocess': 0}
            inference_time = speed_info.get('inference', 0)
            preprocess_time = speed_info.get('preprocess', 0)
            postprocess_time = speed_info.get('postprocess', 0)

             # Extract detections and their labels
            detections = results[0].boxes
            detection_text = []
            for box in detections[:MAX_BOXES_TO_DRAW]:
                class_id = int(box.cls)
                label = model.names[class_id]
                detection_text.append(label)

            detection_count = len(detection_text)
            detection_text_str = ', '.join(detection_text) if detection_text else "no detections"

             # Update the placeholder with the processed frame
            video_placeholder.image(processed_frame[:, :, ::-1], channels="RGB", caption="Video Processing...")
            

            # Display frame processing info
            postprocess_shape = processed_frame.shape 
            st.write(f"Frame {len(processed_frames)}: {frame.shape[0]}x{frame.shape[1]}, "f"(Objects: {detection_count}, detections: {detection_text_str}), "
                     f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, "f"{postprocess_time:.1f}ms postprocess, "f"Processed Frame Shape: {postprocess_shape}")

            processed_frames.append(processed_frame)



        cap.release()
        
        # Save the processed frames as a video
        if save_option == "Yes" and processed_frames:
            result_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                                  (processed_frames[0].shape[1], processed_frames[0].shape[0]))
            for frame in processed_frames:
                out.write(frame)
            out.release()

            st.success("Result video saved successfully!")
            #st.video(result_video_path)

            # Add download buttons
            st.download_button(
                label="Download Result Video",
                data=open(result_video_path, "rb").read(),
                file_name="result_video.mp4",
                mime="video/mp4"
            )

        with open(result_video_path, "rb") as f:
            st.sidebar.title("Detected Video Result")
            st.sidebar.download_button("Download Result Video", f, file_name=result_video_path)


#live webdetection 
elif option == "Livecam Detection":
    st.subheader("Live Webcam Detection")
    
    # Start Webcam Detection button
    run = st.checkbox('Start Webcam Detection')
    
    
    if run:
        # Camera selection
        camera_type = st.radio("Select Camera Type", ["Webcam", "IP Camera"])
        
        if camera_type == "Webcam":
            camera_source = 0  # Default webcam index
        else:
            ip_camera_url = st.text_input("Enter IP Camera URL (rtsp://... or https://...)")
            camera_source = ip_camera_url
        
        if st.button("Connect to Camera"):
            FRAME_WINDOW = st.image([])


            # Add the subheader for Detection Frame Detail Results
            st.subheader("Detection Frame Detail Results")
            history_text = st.empty()
            
            camera = cv2.VideoCapture(camera_source)
            
            if not camera.isOpened():
                st.error("Failed to open the selected camera. Please check your connection.")
            else:
                st.success("Camera connected successfully!")
                
                detection_history = []
                frame_count = 0
                
                while run:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to capture frame from camera. Please check your camera connection.")
                        break
                    
                    start_time = time.time()
                    results = model(frame, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)
                    end_time = time.time()
                    
                    processed_frame = results[0].plot()
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    FRAME_WINDOW.image(rgb_frame, caption="Live Camera Feed")
                    
                    
                    # Extract speed information
                    speed_info = results[0].speed
                    preprocess_time = speed_info.get('preprocess', 0)
                    inference_time = speed_info.get('inference', 0)
                    postprocess_time = speed_info.get('postprocess', 0)

                    # Extract detections and their labels
                    detections = results[0].boxes
                    detection_text = []
                    for box in detections:
                        class_id = int(box.cls)
                        label = model.names[class_id]
                        confidence = float(box.conf)
                        detection_text.append(f"{label} ({confidence:.2f})")
                    

                    detection_count = len(detection_text)
                    detection_text_str = ', '.join(detection_text) if detection_text else "no detections"

                    # Create detection history entry
                    total_time = (end_time - start_time) * 1000  # Convert to ms
                    history_entry = (
                        f"frame:{frame_count}, {frame.shape[0]}x{frame.shape[1]}, "f"(Objects: {detection_count}, detections:({detection_text_str}), {total_time:.1f}ms\n"
                        f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, "
                        f"{postprocess_time:.1f}ms postprocess per image at shape {frame.shape}"
                    )
                    detection_history.append(history_entry)
                    
                    # Display full detection history
                    history_text.text("\n\n".join(detection_history))
                    
                    frame_count += 1
                    
                    # Add a small delay to reduce CPU usage and control frame rate
                    time.sleep(0.1)
                
                camera.release()
    else:
        st.write("Click 'Start Webcam Detection' to begin.")

    st.write("Note: Press 'Stop' in the top right corner or uncheck the 'Start Webcam Detection' box to end the detection.")
