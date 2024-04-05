import os
import shutil
import random

# Base configuration

video_name = 'rpactorpedo'
project_folder = r'C:\Users\Chef\Desktop\HACKERMAN\Programming\Python Projects\yoloTrainer'
background_images_folder = os.path.join(project_folder, 'framesExtracted', video_name)  # Path to your background images
training_data_folder = os.path.join(project_folder, 'TrainingData copy 9')  # Path to your TrainingData folder
labels_folder = os.path.join(project_folder, 'data', video_name)  # Path to your labels for torpedoLab

# Split configuration
train_split = 0.8  # 80% of images go to training
val_split = 0.2  # 20% of images go to validation

# List all background images
background_images = [img for img in os.listdir(background_images_folder) if img.endswith(('.jpg', '.png'))]

# Shuffle the list to ensure random distribution
random.shuffle(background_images)

# Calculate the split index
split_index = int(len(background_images) * train_split)

# Split the images into train and val sets
train_images = background_images[:split_index]
val_images = background_images[split_index:]

# Function to process each subset
def process_images(images, subset):
    subset_folder = os.path.join(subset, video_name)  # Adding subfolder 'torpedoLab' within 'train' and 'val'
    
    for img in images:
        # Define source path for the image
        src_img_path = os.path.join(background_images_folder, img)

        # Define destination paths for the image and label within the subfolder 'torpedoLab'
        dst_img_path = os.path.join(training_data_folder, 'images', subset_folder, img)
        dst_label_path = os.path.join(training_data_folder, 'labels', subset_folder, os.path.splitext(img)[0] + '.txt')

        # Ensure the destination subfolders exist
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)

        # Copy the image to the destination
        shutil.copy(src_img_path, dst_img_path)
        
        # Define source path for the label
        src_label_path = os.path.join(labels_folder, os.path.splitext(img)[0] + '.txt')

        # Check if the label file exists, then copy it
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file for {img} not found.")

        print(f"Processed {img} into {subset_folder}")

# Process training and validation images
process_images(train_images, 'train')
process_images(val_images, 'val')

print("Background images processing complete.")
