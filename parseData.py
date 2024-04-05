import os
import shutil
import random

def rename_labels(source_labels_folder):
    for video_name in os.listdir(source_labels_folder):
        video_folder_path = os.path.join(source_labels_folder, video_name)
        for label_file in os.listdir(video_folder_path):
            frame_number = int(label_file.split('_')[-1].split('.')[0])
            new_label_name = f"{frame_number}.txt"
            src_file_path = os.path.join(video_folder_path, label_file)
            dest_file_path = os.path.join(video_folder_path, new_label_name)
            os.rename(src_file_path, dest_file_path)

def split_data(source_folder, split_size=0.2):
    # Define the structure for images and labels within train and val folders
    subfolders = ['images/train', 'images/val', 'labels/train', 'labels/val']

    # Create the subfolders if they do not exist
    for subfolder in subfolders:
        os.makedirs(os.path.join(source_folder, subfolder), exist_ok=True)

    # List all video names from the images folder
    for video_name in os.listdir(os.path.join(source_folder, 'images')):
        image_folder_path = os.path.join(source_folder, 'images', video_name)
        label_folder_path = os.path.join(source_folder, 'labels', video_name)

        # List all image files and shuffle them for random splitting
        files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
        random.shuffle(files)

        # Determine the split point
        split_point = int(len(files) * (1 - split_size))
        train_files = files[:split_point]
        val_files = files[split_point:]

        # Function to copy files to the respective train/val folders
        def copy_files(files, dest_subfolder):
            for file in files:
                src_image_path = os.path.join(image_folder_path, file)
                src_label_path = os.path.join(label_folder_path, file.replace('.jpg', '.txt'))
                
                dest_image_path = os.path.join(source_folder, 'images', dest_subfolder, video_name, file)
                dest_label_path = os.path.join(source_folder, 'labels', dest_subfolder, video_name, file.replace('.jpg', '.txt'))

                # Make sure the destination directories exist
                os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)

                # Copy the image and, if it exists, the corresponding label
                shutil.copy(src_image_path, dest_image_path)
                if os.path.exists(src_label_path):
                    shutil.copy(src_label_path, dest_label_path)

        # Copy files to their respective directories
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')

main_folder = r'C:\Users\Chef\Desktop\test12'
source_images = os.path.join(main_folder, 'images')
source_labels = os.path.join(main_folder, 'labels')

# Rename label files to match frame naming convention
rename_labels(source_labels)

# Split data into training and validation sets, organizing them into the specified folder structure
split_data(main_folder)
