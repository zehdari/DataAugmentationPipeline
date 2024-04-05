import json
import os

# Path to the COCO JSON file
json_file_path = r'COCOFiles/rpactorpedo.json'

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

# Extract the base name of the JSON file without extension to use as the directory name
json_base_name = os.path.splitext(os.path.basename(json_file_path))[0]

# Directory where the TXT files will be saved, named after the JSON file
output_dir = os.path.join('data', json_base_name)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process annotations
for image in data['images']:
    image_id = image['id']
    img_width, img_height = image['width'], image['height']
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    
    # Adjust file path to include the output directory
    with open(os.path.join(output_dir, f'{image_id}.txt'), 'w') as f:
        for ann in annotations:
            category_id = ann['category_id']
            # Assuming polygons are stored under 'segmentation' and are not RLE encoded
            for coco_polygon in ann['segmentation']:
                yolo_polygon = coco_to_yolo_polygon(coco_polygon, img_width, img_height)
                # Format line: class_id x1 y1 x2 y2 ... xn yn
                line = f"{category_mapping[category_id]} " + " ".join(f"{x} {y}" for x, y in yolo_polygon)
                f.write(line + '\n')
