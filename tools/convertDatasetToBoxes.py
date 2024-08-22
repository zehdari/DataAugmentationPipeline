import os
import glob
import shutil

def convert_polygon_to_bbox(polygon_points):
    x_coords = polygon_points[0::2]
    y_coords = polygon_points[1::2]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Center x, y and width, height
    bbox_x_center = (x_min + x_max) / 2.0
    bbox_y_center = (y_min + y_max) / 2.0
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    return bbox_x_center, bbox_y_center, bbox_width, bbox_height

def copy_and_convert_dataset(original_dataset_path, new_dataset_path):
    labels_path = os.path.join(original_dataset_path, "labels")
    images_path = os.path.join(original_dataset_path, "images")
    
    new_labels_path = os.path.join(new_dataset_path, "labels")
    new_images_path = os.path.join(new_dataset_path, "images")
    
    for subset in ['train', 'val']:
        subset_labels_path = os.path.join(labels_path, subset)
        subset_images_path = os.path.join(images_path, subset)
        
        new_subset_labels_path = os.path.join(new_labels_path, subset)
        new_subset_images_path = os.path.join(new_images_path, subset)
        
        for root, dirs, files in os.walk(subset_labels_path):
            relative_path = os.path.relpath(root, subset_labels_path)
            corresponding_image_folder = os.path.join(subset_images_path, relative_path)
            new_folder_labels_path = os.path.join(new_subset_labels_path, relative_path)
            new_folder_images_path = os.path.join(new_subset_images_path, relative_path)
            
            if not os.path.exists(new_folder_images_path):
                os.makedirs(new_folder_images_path)
            
            # Copy images (handling both .jpg and .png files)
            if os.path.exists(corresponding_image_folder):
                for img_file in glob.glob(os.path.join(corresponding_image_folder, "*.jpg")) + glob.glob(os.path.join(corresponding_image_folder, "*.png")):
                    shutil.copy(img_file, new_folder_images_path)
            
            # Convert and copy labels
            if not os.path.exists(new_folder_labels_path):
                os.makedirs(new_folder_labels_path)
            
            for txt_file in glob.glob(os.path.join(root, "*.txt")):
                with open(txt_file, 'r') as file:
                    lines = file.readlines()
                
                converted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    polygon_points = list(map(float, parts[1:]))
                    
                    bbox = convert_polygon_to_bbox(polygon_points)
                    bbox_str = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
                    converted_lines.append(bbox_str)
                
                # Write the new bbox annotations to the output folder
                new_txt_file = os.path.join(new_folder_labels_path, os.path.basename(txt_file))
                with open(new_txt_file, 'w') as out_file:
                    for bbox_line in converted_lines:
                        out_file.write(bbox_line + '\n')
    
    print(f"Dataset copied and converted. New dataset saved at {new_dataset_path}.")

# Example usage
original_dataset_path = "/Users/cam/Programming/DataAugmentationPipeline/Dataset_Augmented"
new_dataset_path = "/Users/cam/Programming/DataAugmentationPipeline/Dataset_Box_Augmented"
copy_and_convert_dataset(original_dataset_path, new_dataset_path)
