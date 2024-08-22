import os
import shutil
import random

def insert_frames_into_dataset(source_folder, dataset_folder, folder_name_to_insert, train_ratio=0.8, frame_step=1):
    # Paths for the train and val image folders
    train_image_folder = os.path.join(dataset_folder, 'images', 'train', folder_name_to_insert)
    val_image_folder = os.path.join(dataset_folder, 'images', 'val', folder_name_to_insert)

    # Paths for the train and val label folders
    train_label_folder = os.path.join(dataset_folder, 'labels', 'train', folder_name_to_insert)
    val_label_folder = os.path.join(dataset_folder, 'labels', 'val', folder_name_to_insert)

    # Ensure the train and val folders exist
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)

    # Get a list of all frame files in the source folder
    frames = sorted([f for f in os.listdir(source_folder) if f.endswith('.jpg')])

    # Filter frames to include only every `x` frame based on `frame_step`
    frames = frames[::frame_step]

    # Shuffle the frames to randomize the split
    random.shuffle(frames)

    # Calculate the split index
    split_index = int(len(frames) * train_ratio)

    # Split the frames into train and val sets
    train_frames = frames[:split_index]
    val_frames = frames[split_index:]

    # Confirm the split
    print(f"Total frames selected: {len(frames)}")
    print(f"Training frames: {len(train_frames)}")
    print(f"Validation frames: {len(val_frames)}")

    # Copy frames to the train folder and create empty label files
    for frame in train_frames:
        source_path = os.path.join(source_folder, frame)
        destination_image_path = os.path.join(train_image_folder, frame)
        destination_label_path = os.path.join(train_label_folder, frame.replace('.jpg', '.txt'))

        shutil.copy2(source_path, destination_image_path)
        open(destination_label_path, 'w').close()  # Create an empty .txt file
        print(f"Copied {frame} to {train_image_folder} and created {destination_label_path}")

    # Copy frames to the val folder and create empty label files
    for frame in val_frames:
        source_path = os.path.join(source_folder, frame)
        destination_image_path = os.path.join(val_image_folder, frame)
        destination_label_path = os.path.join(val_label_folder, frame.replace('.jpg', '.txt'))

        shutil.copy2(source_path, destination_image_path)
        open(destination_label_path, 'w').close()  # Create an empty .txt file
        print(f"Copied {frame} to {val_image_folder} and created {destination_label_path}")

# Example usage
source_folder = r'/Users/cam/Programming/DataAugmentationPipeline/tools/xframes/rs_thu_background'  # Replace with the path to your source folder
dataset_folder = r'/Users/cam/Programming/DataAugmentationPipeline/Dataset'  # Replace with the path to your dataset folder
folder_name_to_insert = 'woollett_background'  # Name of the folder to insert within train/val
frame_step = 5  # Insert every 5th frame

insert_frames_into_dataset(source_folder, dataset_folder, folder_name_to_insert, frame_step=frame_step)
