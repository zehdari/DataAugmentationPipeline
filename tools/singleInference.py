import os
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('/Users/cam/Programming/DataAugmentationPipeline/tools/models/robosub_2024_v4.pt')

# Set the input and output directories
input_dir = '/Users/cam/Programming/DataAugmentationPipeline/tools/framesExtracted/rs24_thu/'
output_dir = '/Users/cam/Programming/DataAugmentationPipeline/tools/detections'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set the step for processing every x images
x = 51  # Adjust this value as needed

# Initialize counter
counter = 0

# Initialize video writer (set to None initially)
video_writer = None
output_video_path = os.path.join(output_dir, 'output_video.mp4')

# Loop through all the images in the input directory
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more extensions if needed
        if counter % x == 0:
            # Load the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # Perform inference
            results = model(image)
            
            # Check if there are any detections
            if results[0].masks is not None:
                # Parse and display the results
                for i, seg in enumerate(results[0].masks.data):
                    # Convert mask to uint8 before resizing
                    mask = seg.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert resized mask back to boolean
                    mask_resized = mask_resized.astype(bool)
                    
                    # Apply the mask to the image
                    color = np.array([0, 255, 0])  # Green color for segmentation
                    image[mask_resized] = image[mask_resized] * 0.5 + color * 0.5

                    # Get the class name, confidence, and position for the text
                    class_index = int(results[0].boxes.cls[i].item())  # Convert to integer
                    class_name = results[0].names[class_index]
                    confidence = results[0].boxes.conf[i].item()  # Get the confidence score
                    text = f"{class_name} {confidence:.2f}"  # Format the text with class name and confidence
                    position = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]) - 10)

                    # Put class name and confidence text on the image
                    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the output image only if there were detections
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, image)
                print(f"Saved output image with segmentation, class names, and confidence scores to {output_path}")
                
                # Initialize video writer if not already initialized
                if video_writer is None:
                    height, width, _ = image.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
                
                # Write the frame to the video
                video_writer.write(image)
        
        # Increment counter
        counter += 1

# Release the video writer
if video_writer is not None:
    video_writer.release()

print(f"Processing completed! Output video saved to {output_video_path}")
