import cv2
import os

# Specify the path to the folder containing your video files
video_folder_path = r'/Users/cam/Programming/DataAugmentationPipeline/tools/toExtract'
frame_folder_path = r'/Users/cam/Programming/DataAugmentationPipeline/tools/framesExtracted'

# Iterate over each file in the video folder
for video_file in os.listdir(video_folder_path):
    video_path = os.path.join(video_folder_path, video_file)
    
    # Check if the file is a video by its extension
    if video_path.endswith(('.mp4', '.avi', '.mov')):
        # Use OpenCV to read the video
        cap = cv2.VideoCapture(video_path)
        
        # Extract the video name without the extension to create a folder for the frames
        video_name = os.path.splitext(video_file)[0]
        print(f"Extracting {video_name}")
        
        # Create a path for the current video's frames
        current_frame_folder_path = os.path.join(frame_folder_path, video_name)  # Use a different variable here
        
        # Create a folder for the current video's frames if it doesn't already exist
        if not os.path.exists(current_frame_folder_path):
            os.makedirs(current_frame_folder_path)
        
        frame_count = 0
        while True:
            # Read the next frame from the video
            ret, frame = cap.read()
            
            # If the frame was read successfully
            if ret:
                # Construct the filename for the frame
                frame_filename = os.path.join(current_frame_folder_path, f'{frame_count}.jpg')  # Use the new variable here
                
                # Save the frame as an image file
                cv2.imwrite(frame_filename, frame)
                
                frame_count += 1
            else:
                # If no frame is read, it means we are at the end of the video
                break
        
        # Release the VideoCapture object
        cap.release()

print("Frame extraction is complete.")
