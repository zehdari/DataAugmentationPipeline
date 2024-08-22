import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def scan_dataset_for_labels(dataset_folder):
    label_folder = os.path.join(dataset_folder, 'labels')
    class_occurrence = defaultdict(lambda: defaultdict(int))

    for split in ['train', 'val']:
        split_folder = os.path.join(label_folder, split)
        for root, _, files in os.walk(split_folder):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        classes = set()
                        for line in f:
                            class_id = int(line.split()[0])
                            classes.add(class_id)
                        
                        # Update co-occurrence matrix
                        for c1 in classes:
                            for c2 in classes:
                                class_occurrence[c1][c2] += 1

    return class_occurrence

def plot_labels_correlogram(class_occurrence, num_classes):
    matrix = np.zeros((num_classes, num_classes))

    for c1 in range(num_classes):
        for c2 in range(num_classes):
            matrix[c1][c2] = class_occurrence[c1][c2]

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Labels Correlogram')
    plt.xlabel('Class ID')
    plt.ylabel('Class ID')
    plt.show()

# Example usage
dataset_folder = r'/Users/cam/Programming/DataAugmentationPipeline/Dataset'  # Replace with the path to your dataset folder
num_classes = 7  # Replace with the number of classes in your dataset

class_occurrence = scan_dataset_for_labels(dataset_folder)
plot_labels_correlogram(class_occurrence, num_classes)
