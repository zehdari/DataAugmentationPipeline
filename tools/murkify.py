import cv2
import os
import numpy as np

def apply_murky_effect(image):
    # Convert the image to BGR if it is not already
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply a greenish tint
    green_tint = np.full_like(image, (0, 50, 0), dtype=np.uint8)
    murky_image = cv2.addWeighted(image, 0.7, green_tint, 0.3, 0)

    # Add Gaussian blur to simulate murkiness
    blurred_image = cv2.GaussianBlur(murky_image, (15, 15), 0)

    # Add random noise
    noise = np.random.normal(0, 25, blurred_image.shape).astype(np.uint8)
    noisy_image = cv2.add(blurred_image, noise)

    # Reduce contrast
    alpha = 0.5  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    final_image = cv2.convertScaleAbs(noisy_image, alpha=alpha, beta=beta)

    return final_image

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        print(filename)
        if filename.endswith(('.PNG', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(filename)
            # Construct full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(input_path)

            # Apply the murky effect
            murky_image = apply_murky_effect(image)

            # Save the result
            cv2.imwrite(output_path, murky_image)
            print(f"Processed {filename} and saved to {output_path}")

# Parameters
input_folder = '/Users/cam/Downloads/indoor_map_murkified/images'
output_folder = '/Users/cam/Programming/DataAugmentationPipeline/murkified_map'

# Run the augmentation
process_folder(input_folder, output_folder)
