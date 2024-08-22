import os
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('/Users/cam/Programming/DataAugmentationPipeline/tools/models/robosub_2024_v5.pt')

# Set the input and output video paths
input_video_path = '/Users/cam/Programming/DataAugmentationPipeline/tools/toExtract/rs_24_p02.avi'
output_video_path = '/Users/cam/Programming/DataAugmentationPipeline/tools/detections/rs_24_p02_output2.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference on the frame
    results = model(frame)
    
    # Check if there are any detections
    if results[0].masks is not None:
        # Parse and display the results
        for i, seg in enumerate(results[0].masks.data):
            # Convert mask to uint8 before resizing
            mask = seg.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Convert resized mask back to boolean
            mask_resized = mask_resized.astype(bool)
            
            # Apply the mask to the frame
            color = np.array([0, 255, 0])  # Green color for segmentation
            frame[mask_resized] = frame[mask_resized] * 0.5 + color * 0.5

            # Get the class name, confidence, and position for the text
            class_index = int(results[0].boxes.cls[i].item())  # Convert to integer
            class_name = results[0].names[class_index]
            confidence = results[0].boxes.conf[i].item()  # Get the confidence score
            text = f"{class_name} {confidence:.2f}"  # Format the text with class name and confidence
            position = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]) - 10)

            # Put class name and confidence text on the frame
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Write the frame to the output video
    video_writer.write(frame)
    
    # Increment frame counter
    frame_count += 1
    print(f"Processed frame {frame_count}/{total_frames}")
    
# Release the video objects
cap.release()
video_writer.release()

print(f"Processing completed! Output video saved to {output_video_path}")
