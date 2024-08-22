import os

# Define the base directory containing the label files
data_dir = r'C:\Users\Chef\Desktop\HACKERMAN\Programming\Python Projects\yoloTrainer\data'  # Update this to your actual data directory

# Function to rename files with a temporary prefix
def add_temp_prefix(directory, prefix="temp_"):
    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            old_file_path = os.path.join(directory, file_name)
            new_file_path = os.path.join(directory, f"{prefix}{file_name}")
            os.rename(old_file_path, new_file_path)

# Function to remove the prefix and decrement the file index
def remove_temp_prefix_and_decrement(directory, prefix="temp_"):
    for file_name in os.listdir(directory):
        if file_name.startswith(prefix):
            # Remove the prefix and extract the base name
            base_name = file_name[len(prefix):].split('.')[0]
            new_base_name = f"{int(base_name) - 1}.txt"  # Decrement the index
            
            old_file_path = os.path.join(directory, file_name)
            new_file_path = os.path.join(directory, new_base_name)
            
            os.rename(old_file_path, new_file_path)

# Iterate over each subfolder in the data directory
for video_name in os.listdir(data_dir):
    labels_path = os.path.join(data_dir, video_name)

    # Check if the path is a directory before proceeding
    if os.path.isdir(labels_path):
        # Step 1: Add a temporary prefix to each file
        add_temp_prefix(labels_path)

        # Step 2: Remove the prefix and decrement the file index
        remove_temp_prefix_and_decrement(labels_path)

print("Label files have been re-indexed.")
