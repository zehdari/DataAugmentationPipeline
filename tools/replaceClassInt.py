import os
 
# Define the dictionary mapping old integers to new integers
replacement_dict = {
    4: 1,
    5: 2
    # Add more mappings as needed
}
 
# Function to replace the first integer on each line
def replace_first_integer_in_line(line, replacement_dict):
    parts = line.split()
    if parts and parts[0].isdigit():
        old_value = int(parts[0])
        if old_value in replacement_dict:
            parts[0] = str(replacement_dict[old_value])
    return ' '.join(parts)
 
# Function to process files in a directory
def process_directory(directory_path, replacement_dict):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
 
                with open(file_path, 'w') as f:
                    for line in lines:
                        new_line = replace_first_integer_in_line(line, replacement_dict)
                        f.write(new_line + '\n')
 
# Define the root directory containing the labels
root_directory = r'/Users/cam/Programming/DataAugmentationPipeline/output_data/sean_map'
 
# Process train and val directories
for sub_dir in ['train', 'val']:
    directory_path = os.path.join(root_directory, sub_dir)
    process_directory(root_directory, replacement_dict)
 
print("Processing completed!")