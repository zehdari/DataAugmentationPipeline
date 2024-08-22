import json
import os
import shutil
import random
from sklearn.model_selection import train_test_split

def convert_coco_to_yolo(json_file_path, output_dir):
    # Load COCO JSON
    with open(json_file_path) as f:
        data = json.load(f)

    # Assuming 'categories' provides a mapping of category_id to class names or IDs in your YOLOv8 setup
    category_mapping = {cat['id']: cat['name'] for cat in data['categories']}

    # Function to convert COCO polygon to YOLO format
    def coco_to_yolo_polygon(coco_polygon, img_width, img_height):
        # Assuming coco_polygon is a flat list [x1, y1, x2, y2, ..., xn, yn]
        yolo_polygon = []
        for i in range(0, len(coco_polygon), 2):
            x, y = coco_polygon[i], coco_polygon[i+1]
            yolo_polygon.append((x / img_width, y / img_height))
        return yolo_polygon

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process annotations
    for index, image in enumerate(data['images']):
        image_id = image['id']
        img_width, img_height = image['width'], image['height']
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
        
        # Adjust file path to include the output directory
        with open(os.path.join(output_dir, f'{index}.txt'), 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                # Assuming polygons are stored under 'segmentation' and are not RLE encoded
                for coco_polygon in ann['segmentation']:
                    yolo_polygon = coco_to_yolo_polygon(coco_polygon, img_width, img_height)
                    # Format line: class_id x1 y1 x2 y2 ... xn yn
                    line = f"{category_mapping[category_id]} " + " ".join(f"{x} {y}" for x, y in yolo_polygon)
                    f.write(line + '\n')

def copy_and_rename_files_in_order(input_folder_path, output_folder_path, rename=True):
    # Ensure the output directory exists
    os.makedirs(output_folder_path, exist_ok=True)
    
    # List all files in the input folder
    files = os.listdir(input_folder_path)
    
    # Sort files
    files.sort()
    
    # Copy and optionally rename files in sequential order
    for index, file_name in enumerate(files):
        # Construct the old file path
        old_file_path = os.path.join(input_folder_path, file_name)
        
        if rename:
            # Change extension to lowercase .png if it's .PNG
            ext = os.path.splitext(file_name)[1].lower() if os.path.splitext(file_name)[1].lower() == '.png' else os.path.splitext(file_name)[1]
            
            # Construct the new file name and path
            new_file_name = f"{index}{ext}"
        else:
            # Use the original file name
            new_file_name = file_name
        
        new_file_path = os.path.join(output_folder_path, new_file_name)
        
        # Copy the file
        shutil.copy(old_file_path, new_file_path)
        print(f"Copied '{old_file_path}' to '{new_file_path}'")

def replace_labels_in_directory(root_dir, label_mapping):
    # Function to replace text labels in a file with numeric labels
    def replace_labels_in_file(file_path, mapping):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        with open(file_path, 'w') as file:
            for line in lines:
                for text_label, num_label in mapping.items():
                    if line.startswith(text_label):
                        line = line.replace(text_label, num_label, 1)
                        break
                file.write(line)

    # Walk through the directory tree and update files
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(subdir, filename)
                replace_labels_in_file(file_path, label_mapping)

    print("Label conversion completed.")

def split_dataset(folder_list, val_split_ratio=0.2):
    return train_test_split(folder_list, test_size=val_split_ratio, random_state=42)

def process_folder_structure(parent_folder_path, label_output_base='output_data', image_output_base='output_images', dataset_base='Dataset', val_split_ratio=0.2, add_to_dataset=True):
    folder_name = os.path.basename(parent_folder_path)
    
    # Paths for input files
    annotations_folder = os.path.join(parent_folder_path, 'annotations')
    images_folder = os.path.join(parent_folder_path, 'images')
    
    # Paths for output directories
    label_output_dir = os.path.join(label_output_base, folder_name)
    image_output_dir = os.path.join(image_output_base, folder_name)
    
    rename_images = True

    # Check if the folders already exist and skip if they do
    if os.path.exists(image_output_dir):
        print(f"Skipping image processing for {image_output_dir} as it already exists.")
        rename_images = False
    else:
        # Copy and rename images
        copy_and_rename_files_in_order(images_folder, image_output_dir, rename=rename_images)
    
    if os.path.exists(label_output_dir):
        print(f"Skipping label processing for {label_output_dir} as it already exists.")
    else:
        if os.path.exists(annotations_folder):
            # Find the JSON file in the annotations folder
            json_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
            if not json_files:
                raise ValueError(f"No JSON file found in {annotations_folder}")
            json_file_path = os.path.join(annotations_folder, json_files[0])
            
            # Convert annotations
            convert_coco_to_yolo(json_file_path, label_output_dir)
            
            # Define label mapping
            label_mapping = {
                'Buoy': '0', 
                'Mapping_Map': '1', 
                'Mapping_Hole': '2',
                'Gate_Hot': '3', 
                'Gate_Cold': '4',
                'Bin_Temperature': '5',
                'Bin': '6'

            }
            
            # Replace labels in output_data directory
            replace_labels_in_directory(label_output_dir, label_mapping)
        else:
            # Create empty .txt files for each image
            os.makedirs(label_output_dir, exist_ok=True)
            image_files = sorted(os.listdir(images_folder))
            for index, file_name in enumerate(image_files):
                ext = os.path.splitext(file_name)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    open(os.path.join(label_output_dir, f'{index}.txt'), 'w').close()
            print(f"Created empty .txt files for images in {images_folder} as no annotations were found.")
            rename_images = False  # Do not rename images if there are no labels
    
    if rename_images:
        # Rename the images in the output directory if labels were created
        copy_and_rename_files_in_order(images_folder, image_output_dir, rename=True)

    if add_to_dataset:
        # Split the dataset into train and val
        image_files = sorted(os.listdir(image_output_dir))
        label_files = sorted(os.listdir(label_output_dir))
        
        train_images, val_images = split_dataset(image_files, val_split_ratio)
        train_labels, val_labels = split_dataset(label_files, val_split_ratio)
        
        # Define output directories for train and val splits
        train_image_output_dir = os.path.join(dataset_base, 'images', 'train', folder_name)
        val_image_output_dir = os.path.join(dataset_base, 'images', 'val', folder_name)
        train_label_output_dir = os.path.join(dataset_base, 'labels', 'train', folder_name)
        val_label_output_dir = os.path.join(dataset_base, 'labels', 'val', folder_name)
        
        # Ensure the directories exist
        os.makedirs(train_image_output_dir, exist_ok=True)
        os.makedirs(val_image_output_dir, exist_ok=True)
        os.makedirs(train_label_output_dir, exist_ok=True)
        os.makedirs(val_label_output_dir, exist_ok=True)
        
        # Copy files to their respective directories
        for file in train_images:
            if not os.path.exists(os.path.join(train_image_output_dir, file)):
                shutil.copy(os.path.join(image_output_dir, file), os.path.join(train_image_output_dir, file))
        
        for file in val_images:
            if not os.path.exists(os.path.join(val_image_output_dir, file)):
                shutil.copy(os.path.join(image_output_dir, file), os.path.join(val_image_output_dir, file))
        
        for file in train_labels:
            if not os.path.exists(os.path.join(train_label_output_dir, file)):
                shutil.copy(os.path.join(label_output_dir, file), os.path.join(train_label_output_dir, file))
        
        for file in val_labels:
            if not os.path.exists(os.path.join(val_label_output_dir, file)):
                shutil.copy(os.path.join(label_output_dir, file), os.path.join(val_label_output_dir, file))

# Example usage
parent_folder_path = r'/Users/cam/Downloads/woollett_buoy'
process_folder_structure(parent_folder_path, val_split_ratio=0.2, add_to_dataset=True)
