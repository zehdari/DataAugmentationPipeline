import cv2

def trim_video(input_path, output_path, start_frame, end_frame):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure start_frame and end_frame are within bounds
    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        print("Invalid start or end frame.")
        return

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Use 'XVID' for .avi files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames from start_frame to end_frame
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

# Example usage
input_video_path = '/Users/cam/Downloads/rs24_thu.avi'
output_video_path = '/Users/cam/Programming/DataAugmentationPipeline/trimmed_videos/rs_map_far.avi'
start_frame = 9418  # Change to your desired start frame
end_frame = 9500 # Change to your desired end frame

trim_video(input_video_path, output_video_path, start_frame, end_frame)
