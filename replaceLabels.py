import os

# Define the root directory containing your label folders
root_dir = r'/media/ubuntu/USB DISK/TrainingData copy 9/labels/val/rpactorpedo'

# Mapping of text labels to numeric labels
label_mapping = {
    'Buoy': '0', 
    'Glyph1': '1', 
    'Glyph2': '2',
    'Glyph3': '3', 
    'Glyph4': '4', 
    'Gate': '5',
    '5_glyph': '6',
    'Torpedo_open': '7',
    'Torpedo_closed': '8',
    'Torpedo_hole': '9',
    'Bin': '10'

}

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
