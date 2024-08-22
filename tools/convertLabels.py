import os

def replace_text_in_files(folder_path):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Only process text files
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                file_contents = file.readlines()

            # Replace the specified strings in each line
            updated_contents = []
            for line in file_contents:
                updated_line = line.replace("Mapping Map", "Map").replace("Mapping Hole", "Hole")
                updated_contents.append(updated_line)

            # Write the updated contents back to the file
            with open(file_path, 'w') as file:
                file.writelines(updated_contents)

    print("Text replacement completed for all files in the folder.")

# Example usage
folder_path = '/Users/cam/Programming/DataAugmentationPipeline/TrainingData/labels/train/sean_map'
replace_text_in_files(folder_path)
