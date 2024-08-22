import os
import shutil
import cv2

def move_frames(source_folder, destination_folder, ranges, step=1, move_files=True):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    moved_frames = []

    if move_files:
        # Loop through each range and process the frames
        for start_frame, end_frame in ranges:
            for i in range(start_frame, end_frame + 1, step):
                # Construct the filename (assuming the filenames are like '0.jpg', '1.jpg', etc.)
                file_name = f"{i}.jpg"
                source_path = os.path.join(source_folder, file_name)
                destination_path = os.path.join(destination_folder, file_name)

                # Check if the file exists in the source folder before moving
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
                    moved_frames.append(destination_path)
                    print(f"Moved: {file_name}")
                else:
                    print(f"File {file_name} does not exist in the source folder.")
    else:
        # If not moving files, just gather the frame paths in the destination folder
        for file_name in sorted(os.listdir(destination_folder)):
            if file_name.endswith(".jpg"):
                frame_path = os.path.join(destination_folder, file_name)
                moved_frames.append(frame_path)

    return moved_frames

def stitch_frames_to_video(frame_paths, video_path, fps=30):
    if not frame_paths:
        print("No frames to stitch.")
        return

    # Read the first frame to get the width and height
    first_frame = cv2.imread(frame_paths[0])
    height, width, layers = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Add each frame to the video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {video_path}")

# Example usage
source_folder = r"/Users/cam/Programming/DataAugmentationPipeline/tools/framesExtracted/rs24_thu"  # Replace with the path to your source folder
destination_folder = r"/Users/cam/Programming/DataAugmentationPipeline/tools/xframes/rs_thu_background"  # Replace with the path to your destination folder
video_output_path = r"/Users/cam/Programming/DataAugmentationPipeline/tools/xframes/rs_thu_background.avi"  # Replace with the path for the output video

ranges = [
    (0, 4846),
    (7995, 8150),
    (9146, 9229),
    (9525, 9561),
    (11515, 11711),
    (11992, 12315),
    (13250, 13292),
    (13799, 13835),
    (15019, 16359),
    (16400, 16417),
    (17947, 17976),
    (19575, 19682),
    (19858, 19882),
    (19926, 19945),
    (20961, 21222),
    (22242, 22315),
    (24170, 24214),
    (24921, 25460),
    (25515, 25648),
    (28496, 28707),
    (28729, 28733),
    (28764, 28767),
    (28972, 29159)
]

# Set the flag to False if you want to skip moving frames and just stitch together the existing frames in the destination folder
move_files = False

# Move frames (if move_files is True) and get the list of moved frame paths
moved_frames = move_frames(source_folder, destination_folder, ranges, move_files=move_files)

# Stitch the moved frames into a video
stitch_frames_to_video(moved_frames, video_output_path)
