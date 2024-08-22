import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_segmentation_annotations(dataset_folder):
    data = {
        'x_center': [],
        'y_center': [],
        'width': [],
        'height': [],
        'area': []
    }

    for split in ['train', 'val']:
        split_folder = os.path.join(dataset_folder, 'labels', split)
        for folder_name in os.listdir(split_folder):
            folder_path = os.path.join(split_folder, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.txt'):
                        with open(os.path.join(folder_path, file_name), 'r') as f:
                            for line in f:
                                points = list(map(float, line.strip().split()[1:]))
                                x_points = points[0::2]
                                y_points = points[1::2]

                                # Compute bounding box
                                x_min = min(x_points)
                                x_max = max(x_points)
                                y_min = min(y_points)
                                y_max = max(y_points)

                                width = x_max - x_min
                                height = y_max - y_min
                                x_center = (x_min + x_max) / 2
                                y_center = (y_min + y_max) / 2

                                # Calculate the area of the polygon
                                area = 0.5 * np.abs(np.dot(x_points, np.roll(y_points, 1)) - np.dot(y_points, np.roll(x_points, 1)))

                                data['x_center'].append(x_center)
                                data['y_center'].append(y_center)
                                data['width'].append(width)
                                data['height'].append(height)
                                data['area'].append(area)

    return pd.DataFrame(data)

def analyze_polygons(df):
    print("Summary Statistics for Polygon Bounding Boxes and Areas:")
    print(df.describe())

    # Plotting the distribution of polygon center positions
    sns.jointplot(x='x_center', y='y_center', data=df, kind='scatter', alpha=0.5)
    plt.suptitle('Distribution of Polygon Centers')
    plt.show()

    # Plotting the distribution of polygon bounding box dimensions
    sns.jointplot(x='width', y='height', data=df, kind='scatter', alpha=0.5)
    plt.suptitle('Distribution of Polygon Bounding Box Dimensions')
    plt.show()

    # Plotting pairplot for all attributes including area
    sns.pairplot(df, kind="scatter", diag_kind="hist", plot_kws={'alpha':0.5})
    plt.suptitle('Pairplot of Polygon Attributes', y=1.02)
    plt.show()

def check_small_polygons(df, area_threshold=0.001):
    small_polygons = df[df['area'] < area_threshold]
    print(f"\nFound {len(small_polygons)} polygons with area smaller than {area_threshold}:")
    print(small_polygons)

def check_center_bias(df, threshold=0.05):
    center_polygons = df[(df['x_center'] > (0.5 - threshold)) & (df['x_center'] < (0.5 + threshold)) &
                         (df['y_center'] > (0.5 - threshold)) & (df['y_center'] < (0.5 + threshold))]
    print(f"\nFound {len(center_polygons)} polygons centered within {threshold} of the image center:")
    print(center_polygons)

# Example usage
dataset_folder = r'/Users/cam/Programming/DataAugmentationPipeline/Dataset_Augmented_Working'  # Replace with the path to your dataset folder

# Load annotations into a DataFrame
df = load_segmentation_annotations(dataset_folder)

# Analyze polygons
analyze_polygons(df)

# Check for small polygons based on area
check_small_polygons(df, area_threshold=0.001)

# Check for polygons centered around the image center
check_center_bias(df, threshold=0.05)
