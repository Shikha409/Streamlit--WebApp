import streamlit as st
from PIL import Image
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLOv8 model (Adjust path to your model)
model = YOLO("weights/best.pt")

# Streamlit app
st.title("YOLOv8 Objects Detection App")


# Sidebar for options
st.sidebar.title("Detection Options Config")
option = st.sidebar.radio("1.Select Input Type:", ("Upload Image", "Upload Video", "Webcam Detection"))

# Sidebar settings
MAX_BOXES_TO_DRAW = st.sidebar.number_input('2.Maximum Boxes To Draw', value=5, min_value=1, max_value=5)
deviceLst = ['cpu', '0', '1', '2', '3']
DEVICES = st.sidebar.selectbox("3.Select Device", deviceLst, index=0)
print("Devices: ", DEVICES)  # Print selected device for debugging purposes
MIN_SCORE_THRES = st.sidebar.slider('4.Min Confidence Score Threshold', min_value=0.0, max_value=1.0, value=0.4)
enable_cpu = st.sidebar.checkbox("5.Enable CPU Processing", value=True)
save_option = st.sidebar.selectbox("6.Save Result?", ("Yes", "No"))

# Initialize paths for saving results
result_image_path = "result_image.jpg"
result_video_path = "result_video.mp4"

# Set the model to use the selected device
model.to(DEVICES)

# Image upload processing
if option == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("7.Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Display the uploaded image and detected image in the same row
        col1, col2 = st.columns(2)

        with col1:
            st.write("Test for Objects")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("Detected objects result")
            # Run model inference on the uploaded image
            results = model(image, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)

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

            # Display the detected image
            detected_image = results[0].plot()[:, :, ::-1]
            st.image(detected_image, caption="Detected Image", use_column_width=True)

            # Save and display the detected image result
            if save_option == "Yes":
                results[0].save(result_image_path)
                st.success(f"Result saved as {result_image_path}")
                with open(result_image_path, "rb") as f:
                    st.sidebar.title("8.Detected Image Result")
                    st.sidebar.download_button("Download Result Image", f, file_name=result_image_path)

        st.subheader("Detection Frame Detail Results")
        
        st.write(f"Image Details: {image.size[1]}x{image.size[0]}, (Objects:{detection_count} , detections: {detection_text_str})")
        st.write(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess")

# Function to load YOLOR and process each frame with DeepSORT
def load_yolor_and_process_each_frame(frame, min_score_thresh, enable_cpu):
     results = model(frame, conf=min_score_thresh, max_det=MAX_BOXES_TO_DRAW)
     return results

        
# Video upload processing
if option == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("7.Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        video_path = uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.subheader("Upload Tested Video ")
        st.video(video_path)  # Show uploaded video before detection

        # Video processing
        #st.write("Upload Tested Video ")
        cap = cv2.VideoCapture(video_path)
        processed_frames = []

        # Create a placeholder for the video
        video_placeholder = st.empty()

        st.subheader("Result Will Be Save Here After Detection")
        with open(result_video_path, "rb") as f:
         st.download_button(f" saved as {result_video_path}"
                   "Download Result Video", f, file_name=result_video_path)
        
        
        # Display "Detection Frame Detail Results" title in the middle of the page
        st.subheader("Detection Frame Detail Results")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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
        os.remove(video_path)  # Clean up the uploaded video

        # Save the processed frames as a video
        if save_option == "Yes" and processed_frames:
            out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (processed_frames[0].shape[1], processed_frames[0].shape[0]))
            for frame in processed_frames:
                out.write(frame)
            out.release()
            st.success(f"Result video saved as {result_video_path}")

        # Show final processed video and provide download option
        st.video(result_video_path)
        with open(result_video_path, "rb") as f:
            st.sidebar.title("8.Detected Video Result")
            st.sidebar.download_button("Download Result Video", f, file_name=result_video_path)

        #st.text("Video is Processed")  # Indicate processing completion



# Webcam Detection
elif option == "Webcam Detection":
    start_button = st.button("Start Webcam Detection")

    if start_button:
        cap = cv2.VideoCapture(0)
        processed_frames = []

        # Create a placeholder for the video
        video_placeholder = st.empty()

        st.subheader("Detection Frame Detail Results")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=MIN_SCORE_THRES, max_det=MAX_BOXES_TO_DRAW)
            processed_frame = results[0].plot()
            processed_frames.append(processed_frame)

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
            video_placeholder.image(processed_frame[:, :, ::-1], channels="RGB", caption="Webcam Detection")

            # Display frame processing info 
            st.write(f"Frame {len(processed_frames)-1}: {frame.shape[0]}x{frame.shape[1]}, (Objects:{detection_count}, {detection_text_str}), "
                     f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess,")

        cap.release()

        # Save the processed webcam video
        if save_option == "Yes" and processed_frames:
            out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (processed_frames[0].shape[1], processed_frames[0].shape[0]))
            for frame in processed_frames:
                out.write(frame)
            out.release()
