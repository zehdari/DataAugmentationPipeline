
# Dataset Augmentation Tool

This tool provides a graphical interface for augmenting image datasets with various techniques such as mirroring, cropping, zooming, rotating, and overlaying additional images. Users can preview the augmentations and apply them to individual images or the entire dataset.

![Dataset Augmentation Tool](assets/dataset_augmentation_tool.png)

## Features

- **Directory Selection**: Choose the dataset root and overlay image directory.
- **Augmentation Configuration**: Adjust parameters like mirror weights, crop weights, zoom weights, rotate weights, and overlay weights.
- **Skip Augmentations**: Option to skip certain augmentations for specific folders.
- **Image Viewer**: View and navigate through images in the dataset.
- **Single Image Augmentation**: Augment the currently displayed image.
- **Batch Augmentation**: Apply augmentations to the entire dataset.

## Setting Up the Environment

1. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3. **Install the Required Dependencies**:
    ```bash
    pip install PyQt5 opencv-python-headless numpy shapely
    ```

## Usage

1. **Initialize and Run**:
    ```bash
    python augment_gui.py
    ```

2. **Select Directories**: Use the buttons to select the dataset root directory and overlay image directory.
3. **Configure Augmentations**: Adjust the augmentation parameters using the sliders and checkboxes.
4. **Preview and Navigate**: Use the image viewer to navigate through the dataset and preview the augmentations.
5. **Single Image Augmentation**: Click "Augment Current Image" to apply augmentations to the currently displayed image.
6. **Batch Augmentation**: Click "Run Augmentation" to apply the configured augmentations to the entire dataset.

## Running the Application

To run the application, execute the script:
```bash
python augment_gui.py
```

The application window will open, allowing you to configure and apply augmentations to your dataset.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
