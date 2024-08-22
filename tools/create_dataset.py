import os
import shutil
import random

# Define paths
output_images_path = '/Users/cam/Programming/DataAugmentationPipeline/output_images'
output_labels_path = '/Users/cam/Programming/DataAugmentationPipeline/output_data'
dataset_path = '/Users/cam/Programming/DataAugmentationPipeline/Dataset'

# Define the train/val split ratio
train_ratio = 0.8

# Ensure the dataset directory structure
os.makedirs(os.path.join(dataset_path, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'labels', 'val'), exist_ok=True)

# Function to move files to the corresponding dataset folder
def move_files(src_folder, dest_folder, files, is_label=False):
    for file in files:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder, file)
        
        # If it's a label, check if it exists; if not, create an empty file
        if is_label:
            if not os.path.exists(src_path):
                open(dest_path, 'w').close()
            else:
                shutil.copy2(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

# Process each folder in output_images and output_labels
for folder_name in os.listdir(output_images_path):
    image_folder = os.path.join(output_images_path, folder_name)
    label_folder = os.path.join(output_labels_path, folder_name)
    
    if not os.path.isdir(image_folder) or not os.path.isdir(label_folder):
        continue
    
    # Create corresponding directories in the dataset
    os.makedirs(os.path.join(dataset_path, 'images', 'train', folder_name), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'images', 'val', folder_name), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'labels', 'train', folder_name), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'labels', 'val', folder_name), exist_ok=True)
    
    # List all files in the image folder
    image_files = os.listdir(image_folder)
    
    # Shuffle files to ensure randomness
    random.shuffle(image_files)
    
    # Calculate the split index
    split_index = int(len(image_files) * train_ratio)
    
    # Split files into train and val sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    # Move files to the corresponding train/val folders
    move_files(image_folder, os.path.join(dataset_path, 'images', 'train', folder_name), train_files)
    move_files(image_folder, os.path.join(dataset_path, 'images', 'val', folder_name), val_files)
    
    # Move label files, creating empty ones if they don't exist
    train_label_files = [f.replace(os.path.splitext(f)[1], '.txt') for f in train_files]
    val_label_files = [f.replace(os.path.splitext(f)[1], '.txt') for f in val_files]
    
    move_files(label_folder, os.path.join(dataset_path, 'labels', 'train', folder_name), train_label_files, is_label=True)
    move_files(label_folder, os.path.join(dataset_path, 'labels', 'val', folder_name), val_label_files, is_label=True)

print("Files have been successfully split, and empty label files created where necessary.")
