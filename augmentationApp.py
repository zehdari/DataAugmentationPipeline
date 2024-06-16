import sys
import os
import re
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QSlider, QSpinBox, QMessageBox, QGroupBox, QFormLayout,
                             QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QGridLayout, QSplitter,
                             QListWidget, QListWidgetItem, QSizePolicy, QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt, QObject, QEvent, QSize, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QImage
import cv2
from augment_data import augment_image
import numpy as np

class ClickFilter(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            for widget in obj.findChildren(QLineEdit):
                widget.clearFocus()
        return super().eventFilter(obj, event)

class CustomLineEdit(QLineEdit):
    def focusOutEvent(self, event):
        self.deselect()
        super().focusOutEvent(event)

class AugmentDatasetGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.dataset_root = ""
        self.overlay_image_dir = ""
        self.output_dir = ""
        self.skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }
        self.image_paths = []  # List to store image paths
        self.folder_images = []
        self.label_paths = {}  # Dictionary to store label paths
        self.current_image_index = 0  # To store the index of the current displayed image
        self.current_image_path = ""
        self.folder_name = ""
        self.class_colors = {}  # Dictionary to store class colors
        self.augmented_image = None
        self.augmented_polygons = None
        self.augmented_image_original_dims = None

        self.rotation_random_vs_90 = [25, 75]
        self.zoom_in_vs_out_weights = [40, 60]
        self.zoom_in_min_padding = 0.05
        self.zoom_in_max_padding = 0.5
        self.zoom_out_min_padding = 0.1
        self.zoom_out_max_padding = 0.8
        self.zoom_padding = [self.zoom_in_min_padding, self.zoom_in_max_padding, self.zoom_out_min_padding, self.zoom_out_max_padding]
        self.maintain_aspect_ratio_weights = [50, 50]
        self.overlay_min_max_scale = [0.3, 1.0]
        self.overlay_scale_weights = [98, 2]

        self.show_labels = True
        self.show_polygons = True 
        self.show_bounding_boxes = False 
        self.show_points = False 

        self.output_dir_set = False  # Flag to track if output directory has been set

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dataset Augmentation GUI')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        tab_widget = QTabWidget()
        self.augmentation_settings_tab = QWidget()
        self.image_viewer_tab = QWidget()

        tab_widget.addTab(self.augmentation_settings_tab, "Augmentation Settings")
        tab_widget.addTab(self.image_viewer_tab, "Image Viewer")

        self.init_augmentation_settings_tab()
        self.init_image_viewer_tab()

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

        # Install event filter
        self.installEventFilter(ClickFilter(self))

    def init_augmentation_settings_tab(self):
        layout = QVBoxLayout()

        # Combined directory selection
        dir_group = QGroupBox("Select Directories")
        dir_layout = QFormLayout()
        self.dataset_label = QLabel("Not selected")
        self.overlay_label = QLabel("Not selected")
        self.output_dir_label = QLabel("Not selected")
        self.dataset_btn = QPushButton("Select Dataset Root")
        self.overlay_btn = QPushButton("Select Overlay Image Directory")
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.dataset_btn.clicked.connect(self.select_dataset_root)
        self.overlay_btn.clicked.connect(self.select_overlay_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        dir_layout.addRow(self.dataset_btn, self.dataset_label)
        dir_layout.addRow(self.overlay_btn, self.overlay_label)
        dir_layout.addRow(self.output_dir_btn, self.output_dir_label)
        dir_group.setLayout(dir_layout)
        dir_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(dir_group)

        # Sliders and Skip Augmentations
        weights_skip_layout = QSplitter(Qt.Horizontal)
        
        # Weights sliders with scroll area inside a group box
        weights_group = QGroupBox("Augmentation Settings")
        weights_layout = QGridLayout()

        self.mirror_slider, self.mirror_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Mirror Weights (Mirror/No Mirror):", self.mirror_slider, self.mirror_value, 0)

        self.crop_slider, self.crop_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Crop Weights (Crop/No Crop):", self.crop_slider, self.crop_value, 1)

        self.zoom_slider, self.zoom_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Zoom Weights (Zoom/No Zoom):", self.zoom_slider, self.zoom_value, 2)

        self.rotate_slider, self.rotate_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Rotate Weights (Rotate/No Rotate):", self.rotate_slider, self.rotate_value, 3)

        self.overlay_slider, self.overlay_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Overlay Weights (Overlay/No Overlay):", self.overlay_slider, self.overlay_value, 4)

        self.rotation_random_vs_90_slider, self.rotation_random_vs_90_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Rotation Random vs 90: ", self.rotation_random_vs_90_slider, self.rotation_random_vs_90_value, 5)

        self.zoom_in_vs_out_slider, self.zoom_in_vs_out_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Zoom In vs Out Weights: ", self.zoom_in_vs_out_slider, self.zoom_in_vs_out_value, 6)

        self.maintain_aspect_ratio_slider, self.maintain_aspect_ratio_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Maintain Aspect Ratio Weights: ", self.maintain_aspect_ratio_slider, self.maintain_aspect_ratio_value, 7)

        self.overlay_scale_slider, self.overlay_scale_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Overlay Scale Weights: ", self.overlay_scale_slider, self.overlay_scale_value, 8)

        weights_group.setLayout(weights_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(weights_layout)
        scroll_area.setWidget(scroll_widget)

        group_box_with_scroll = QGroupBox("Augmentation Settings")
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(scroll_area)
        group_box_with_scroll.setLayout(group_box_layout)
        group_box_with_scroll.setMinimumWidth(400)

        weights_skip_layout.addWidget(group_box_with_scroll)

        # Skip Augmentations
        self.skip_group = QGroupBox("Skip Augmentations for Folders")
        self.skip_layout = QVBoxLayout()
        self.skip_table = QTableWidget()
        self.skip_table.setColumnCount(7)  # Update the column count
        self.skip_table.setHorizontalHeaderLabels(['Folder', 'Zoom', 'Crop', 'Rotate', 'Mirror', 'Overlay', 'Skip All'])  # Add new header
        self.skip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 7):
            self.skip_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Fixed)
            self.skip_table.setColumnWidth(col, 50)
        self.skip_layout.addWidget(self.skip_table)
        self.skip_group.setLayout(self.skip_layout)
        weights_skip_layout.addWidget(self.skip_group)

        weights_skip_layout.setSizes([800, 400])  # Initial sizes of the panels, evenly split
        weights_skip_layout.setCollapsible(0, False)
        weights_skip_layout.setCollapsible(1, False)

        # Set minimum sizes
        self.skip_group.setMinimumWidth(400)  # Set a minimum width for the skip group
        weights_skip_layout.setMinimumWidth(1200)  # Ensure the splitter does not resize smaller than the initial setup
        weights_skip_layout.setHandleWidth(10)

        layout.addWidget(weights_skip_layout)
        self.augmentation_settings_tab.setLayout(layout)

    def init_image_viewer_tab(self):
        layout = QVBoxLayout()

        # Image viewer layout
        self.image_viewer_layout = QVBoxLayout()

        folder_list_layout = QHBoxLayout()
        self.folder_list = QListWidget()
        self.folder_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy to Expanding
        self.folder_list.setMaximumHeight(125)  # Set maximum height for the folder list
        self.folder_list.setMinimumHeight(50)  # Set minimum height for the folder list
        self.folder_list.itemClicked.connect(self.display_images)
        folder_list_layout.addWidget(self.folder_list)

        checkboxes_button_layout = QVBoxLayout()

        self.labels_checkbox = QCheckBox("Show Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self.toggle_labels)
        checkboxes_button_layout.addWidget(self.labels_checkbox)

        self.polygons_checkbox = QCheckBox("Show Polygons")
        self.polygons_checkbox.setChecked(True)
        self.polygons_checkbox.stateChanged.connect(self.toggle_polygons)
        checkboxes_button_layout.addWidget(self.polygons_checkbox)

        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(False)
        self.bbox_checkbox.stateChanged.connect(self.toggle_bounding_boxes)
        checkboxes_button_layout.addWidget(self.bbox_checkbox)

        self.points_checkbox = QCheckBox("Show Points")
        self.points_checkbox.setChecked(False)
        self.points_checkbox.stateChanged.connect(self.toggle_points)
        checkboxes_button_layout.addWidget(self.points_checkbox)

        self.augment_single_btn = QPushButton("Preview Augmentation")
        self.augment_single_btn.clicked.connect(self.augment_current_image)
        checkboxes_button_layout.addWidget(self.augment_single_btn)

        # Add button to save current preview
        self.save_preview_btn = QPushButton("Save Current Preview")
        self.save_preview_btn.clicked.connect(self.save_current_preview)
        checkboxes_button_layout.addWidget(self.save_preview_btn)

        folder_list_layout.addLayout(checkboxes_button_layout)

        folder_list_layout.setStretch(0, 1)  # Set stretch factor for folder_list
        folder_list_layout.setStretch(1, 0)  # Adjust margins as needed

        self.image_viewer_layout.addLayout(folder_list_layout)

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_viewer_layout.addWidget(self.image_name_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumHeight(150)
        self.image_viewer_layout.addWidget(self.image_label)

        # Navigation controls
        self.image_navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.valueChanged.connect(self.slider_value_changed)
        self.image_navigation_layout.addWidget(self.prev_button)
        self.image_navigation_layout.addWidget(self.image_slider)
        self.image_navigation_layout.addWidget(self.next_button)

        self.image_viewer_layout.addLayout(self.image_navigation_layout)

        layout.addLayout(self.image_viewer_layout)

        # Augment and save button
        self.augment_and_save_btn = QPushButton("Augment and Save Current Image")
        self.augment_and_save_btn.clicked.connect(self.augment_and_save_current_image)
        layout.addWidget(self.augment_and_save_btn)

        # Run button
        self.run_btn = QPushButton("Run Augmentation")
        self.run_btn.clicked.connect(self.run_augmentation)
        layout.addWidget(self.run_btn)

        self.image_viewer_tab.setLayout(layout)

    def create_slider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setEnabled(False)
        value_label = CustomLineEdit("50")
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setEnabled(False)
        slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
        value_label.textChanged.connect(lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None)
        return slider, value_label

    def add_slider_to_layout(self, layout, label_text, slider, value_label, row):
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

    def select_dataset_root(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if dir_name:
            self.dataset_root = dir_name
            self.dataset_label.setText(dir_name)
            if not self.output_dir_set:
                self.prompt_for_output_dir()
            self.update_sliders_state()
            self.scan_folders()

    def select_overlay_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Overlay Image Directory")
        if dir_name:
            self.overlay_image_dir = dir_name
            self.overlay_label.setText(dir_name)
            self.update_sliders_state()

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_label.setText(dir_name)
        else:
            self.prompt_for_output_dir()

    def prompt_for_output_dir(self):
        while not self.output_dir:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Output Directory")
            msg_box.setText(f"Would you like to specify an output directory? \n Default: {self.dataset_root}_Augmented")
            specify_btn = msg_box.addButton("Specify", QMessageBox.AcceptRole)
            default_btn = msg_box.addButton("Default", QMessageBox.RejectRole)
            msg_box.exec_()

            if msg_box.clickedButton() == specify_btn:
                dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if dir_name:
                    self.output_dir = dir_name
                    self.output_dir_label.setText(dir_name)
            else:
                self.output_dir = self.dataset_root + "_Augmented"
                self.output_dir_label.setText(self.output_dir)
            self.output_dir_set = True

    def scan_folders(self):
        # Clear previous skip augmentation inputs
        for key in self.skip_augmentations.keys():
            self.skip_augmentations[key] = []

        # Scan dataset for folders and images
        folders = set()
        self.image_paths = []  # Reset image paths
        self.label_paths = {}  # Reset label paths

        for root, dirs, files in os.walk(self.dataset_root):
            if os.path.basename(root).lower() not in ['images', 'labels']:
                for name in dirs:
                    if name.lower() not in ['train', 'val', 'labels', 'images']:
                        folders.add(name)
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
                        label_path = os.path.join(self.dataset_root, 'labels', os.path.relpath(image_path, os.path.join(self.dataset_root, 'images')).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                        self.label_paths[image_path] = label_path

        folders = list(folders)
        folders.sort()

        self.skip_table.setRowCount(len(folders))

        self.folder_list.clear()
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            list_item = QListWidgetItem(folder)
            self.folder_list.addItem(list_item)
            for col in range(1, 7):  # Update the range to include the new column
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left 
                if col == 5:
                    checkbox.setEnabled(False)
                if col == 6:  # Connect the new checkbox to the slot
                    checkbox.stateChanged.connect(lambda state, r=row: self.toggle_skip_all(state, r))
                self.skip_table.setCellWidget(row, col, checkbox)

        # Sort images numerically
        self.image_paths.sort(key=self.natural_keys)

    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.Checked
        for col in range(1, 6):  # Update to check relevant columns
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
        if not self.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(False)

    def display_images(self, item):
        self.folder_name = item.text()
        self.folder_images = [path for path in self.image_paths if self.folder_name in path]
        self.image_slider.setMaximum(len(self.folder_images) - 1)
        self.current_image_index = 0
        self.augmented_image = None  # Clear the augmented image when folder changes
        self.show_image()

    def show_image(self):
        if self.augmented_image is not None:
            # Display the augmented image
            self.image_name_label.setText(f"(Preview) {os.path.basename(self.current_image_path)}")
            self.display_image_and_polygons(self.augmented_image, self.augmented_polygons)
        elif self.folder_images:
            self.current_image_path = self.folder_images[self.current_image_index]

            # Check if the augmented image exists
            relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
            augmented_image_path = os.path.join(self.output_dir, relative_image_path)

            if os.path.exists(augmented_image_path):
                image = cv2.imread(augmented_image_path)
                self.image_name_label.setText(f"(Augmented) {os.path.basename(self.current_image_path)}")
            else:
                image = cv2.imread(self.current_image_path)
                self.image_name_label.setText(os.path.basename(self.current_image_path))

            # Check if the augmented label exists
            label_path = self.label_paths.get(self.current_image_path)
            relative_label_path = os.path.relpath(label_path, self.dataset_root)
            augmented_label_path = os.path.join(self.output_dir, relative_label_path)

            if os.path.exists(augmented_label_path):
                polygons, labels = self.load_polygons_and_labels(augmented_label_path, image.shape)
            else:
                polygons, labels = self.load_polygons_and_labels(label_path, image.shape)

            self.display_image_and_polygons(image, polygons)
            self.update_navigation_buttons()

    def update_sliders_state(self):
        enable_normal_sliders = bool(self.dataset_root)
        enable_overlay_sliders = bool(self.overlay_image_dir)
        
        # Enable/disable normal sliders
        self.mirror_slider.setEnabled(enable_normal_sliders)
        self.mirror_value.setEnabled(enable_normal_sliders)
        self.crop_slider.setEnabled(enable_normal_sliders)
        self.crop_value.setEnabled(enable_normal_sliders)
        self.zoom_slider.setEnabled(enable_normal_sliders)
        self.zoom_value.setEnabled(enable_normal_sliders)
        self.rotate_slider.setEnabled(enable_normal_sliders)
        self.rotate_value.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_slider.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_value.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_slider.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_value.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_slider.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_value.setEnabled(enable_normal_sliders)
        
        # Enable/disable overlay sliders
        self.overlay_slider.setEnabled(enable_overlay_sliders)
        self.overlay_value.setEnabled(enable_overlay_sliders)
        self.overlay_scale_slider.setEnabled(enable_overlay_sliders)
        self.overlay_scale_value.setEnabled(enable_overlay_sliders)

        for row in range(self.skip_table.rowCount()):
            overlay_checkbox = self.skip_table.cellWidget(row, 5)  # 5 is the index of the Overlay column
            overlay_checkbox.setEnabled(enable_overlay_sliders)

    def toggle_bounding_boxes(self, state):
        self.show_bounding_boxes = state == Qt.Checked
        self.show_image()

    def toggle_polygons(self, state):
        self.show_polygons = state == Qt.Checked
        self.show_image()

    def toggle_labels(self, state):
        self.show_labels = state == Qt.Checked
        self.show_image()

    def toggle_points(self, state):
        self.show_points = state == Qt.Checked
        self.show_image()

    def display_image_and_polygons(self, image, polygons):
        height, width, _ = image.shape
        image_bytes = image.tobytes()
        qimage = QImage(image_bytes, width, height, width * 3, QImage.Format_RGB888)
        qimage = qimage.rgbSwapped()  # Convert BGR to RGB

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(scaled_pixmap)

        # Enable anti-aliasing for sharper lines and text
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        labels = []  # To store labels and positions for later drawing

        if self.augmented_image is not None and self.augmented_image_original_dims is not None:
            orig_h, orig_w = self.augmented_image_original_dims
        else:
            orig_h, orig_w = height, width

        for polygon in polygons:
            class_id = polygon[0]
            if class_id not in self.class_colors:
                self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Set pen for polygon lines and bounding boxes to full opacity
            pen = QPen(self.class_colors[class_id], 2)
            pen_color = self.class_colors[class_id]
            pen.setColor(pen_color)

            # Set brush color with desired opacity
            brush_color = self.class_colors[class_id]
            brush_color.setAlpha(100)  # Set fill opacity here (0-255)
            brush = QBrush(brush_color)
            brush.setStyle(Qt.SolidPattern)

            points = [QPointF(pt[0] * scaled_pixmap.width() / orig_w, pt[1] * scaled_pixmap.height() / orig_h) for pt in polygon[1:]]

            if self.show_polygons:
                pen_color.setAlpha(100)
                pen = QPen(pen_color, 2)
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPolygon(*points)

            if self.show_points:
                for point in points:
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(point, 2.5, 2.5)
                    painter.setPen(QPen(self.class_colors[class_id], 1))
                    painter.drawEllipse(point, 1.5, 1.5)

            # Calculate bounding box
            min_x = min(point.x() for point in points)
            max_x = max(point.x() for point in points)
            min_y = min(point.y() for point in points)
            max_y = max(point.y() for point in points)

            if self.show_bounding_boxes:
                # Draw bounding box with full opacity
                bounding_box_pen = QPen(self.class_colors[class_id], 1)
                bounding_box_pen_color = self.class_colors[class_id]
                bounding_box_pen_color.setAlpha(255)
                bounding_box_pen.setColor(bounding_box_pen_color)

                painter.setPen(bounding_box_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))

            # Store label information for later drawing
            labels.append((class_id, points[0]))

        # Draw all labels
        if self.show_labels:
            for class_id, position in labels:
                # Draw text with black outline
                painter.setPen(QPen(Qt.black, 2))
                font_metrics = painter.fontMetrics()
                text_height = font_metrics.height()
                text_width = font_metrics.horizontalAdvance(class_id)
                label_position = position + QPointF(0, text_height)
                
                # Adjust label position to prevent going off the canvas
                if label_position.x() + text_width > scaled_pixmap.width():
                    label_position.setX(scaled_pixmap.width() - text_width/2)
                if label_position.y() > scaled_pixmap.height():
                    label_position.setY(scaled_pixmap.height() - text_height/2)

                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    painter.drawText(label_position + QPointF(dx, dy), class_id)

                # Draw text in white on top
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(label_position, class_id)

        painter.end()

        self.image_label.setPixmap(scaled_pixmap)

    def convert_bbox_to_polygon(self, bbox):
        class_id = bbox[0]
        x_center, y_center, width, height = map(float, bbox[1:])
        half_w = width / 2
        half_h = height / 2
        points = [
            x_center - half_w, y_center - half_h,
            x_center + half_w, y_center - half_h,
            x_center + half_w, y_center + half_h,
            x_center - half_w, y_center + half_h
        ]
        return [class_id] + points

    def load_polygons_and_labels(self, label_path, target_size):
        polygons = []

        if (label_path and os.path.exists(label_path)):
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            for line in label_data:

                line_data = line.strip().split()
                annotation_type = self.identify_annotation_type(line_data)
                if (annotation_type == 'bbox'):
                    polygon_data = self.convert_bbox_to_polygon(line_data)
                else:
                    polygon_data = line_data

                if len(polygon_data) < 5:
                    continue  # Ensure there are enough coordinates for a polygon

                class_id = polygon_data[0]
                if class_id not in self.class_colors:
                    self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                pen = QPen(self.class_colors[class_id], 2)
                brush_color = self.class_colors[class_id]
                brush_color.setAlpha(100)  # Set opacity here (0-255)
                brush = QBrush(brush_color)
                brush.setStyle(Qt.SolidPattern)

                # Extract normalized coordinates
                coords = list(map(float, polygon_data[1:]))
                points = [(coords[i] * target_size[1], coords[i+1] * target_size[0]) for i in range(0, len(coords), 2)]

                polygons.append([class_id] + points)

        return polygons, []

    def identify_annotation_type(self, parts):
        
        if len(parts) < 5:
            return "unknown"
        if len(parts) % 2 == 1 and len(parts) > 5:
            return "polygon"
        elif len(parts) == 5:
            return "bbox"
        else:
            return "unknown"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.folder_images) - 1)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.folder_images) - 1:
            self.current_image_index += 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def slider_value_changed(self, value):
        if value != self.current_image_index:
            self.current_image_index = value
            self.augmented_image = None
            self.show_image()

    def run_augmentation(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        if not self.output_dir_set:
            self.prompt_for_output_dir()

        # Add the logic to run the augmentation for all images
        pass

    def augment_current_image(self):
        if not self.dataset_root or not self.overlay_image_dir:
            overlay_weights = [0, 100]
        else:
            overlay_weights = [self.overlay_slider.value(), 100 - self.overlay_slider.value()]
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Input Required", "Please select an image to augment.")
            return
        
        mirror_weights = [self.mirror_slider.value(), 100 - self.mirror_slider.value()]
        crop_weights = [self.crop_slider.value(), 100 - self.crop_slider.value()]
        zoom_weights = [self.zoom_slider.value(), 100 - self.zoom_slider.value()]
        rotate_weights = [self.rotate_slider.value(), 100 - self.rotate_slider.value()]
        overlay_scale_weights = [self.overlay_scale_slider.value(), 100 - self.overlay_scale_slider.value()]
        maintain_aspect_ratio_weights = [self.maintain_aspect_ratio_slider.value(), 100 - self.maintain_aspect_ratio_slider.value()]
        zoom_in_vs_out_weights = [self.zoom_in_vs_out_slider.value(), 100 - self.zoom_in_vs_out_slider.value()]

        # Load the image
        image = cv2.imread(self.current_image_path)
        (h, w) = image.shape[:2]

        # Load the label file if it exists
        label_path = self.label_paths.get(self.current_image_path)
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                polygons = []
                class_ids = []
                for line in lines:
                    parts = line.strip().split()
                    if self.identify_annotation_type(parts) == 'bbox':
                        parts = self.convert_bbox_to_polygon(parts)
                    
                    class_id = parts[0]
                    class_ids.append(class_id)
                    polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                    polygons.append(polygon)
        else:
            polygons = []
            class_ids = []

        # Run the augment_image function
        augmented_image, augmented_polygons = augment_image(
            image,
            polygons,
            self.folder_name,
            class_ids,
            h,
            w,
            self.skip_augmentations, 
            mirror_weights, 
            crop_weights,
            overlay_weights, 
            overlay_scale_weights, 
            self.overlay_min_max_scale,
            maintain_aspect_ratio_weights, 
            zoom_weights, 
            zoom_in_vs_out_weights,
            self.zoom_padding,
            self.overlay_image_dir if self.overlay_image_dir else ""
        )

        (new_h, new_w) = augmented_image.shape[:2]

        denormalized_polygons = []
        for polygon in augmented_polygons:
            class_id = polygon[0]
            denormalized_polygon = [class_id] + [(int(x * new_w), int(y * new_h)) for (x, y) in polygon[1:]]
            denormalized_polygons.append(denormalized_polygon)

        self.augmented_image = augmented_image
        self.augmented_polygons = denormalized_polygons
        self.augmented_image_original_dims = (new_h, new_w)
        self.show_image()

    def augment_and_save_current_image(self):
        self.augment_current_image()
        self.save_augmented_image()

    def save_current_preview(self):
        if self.augmented_image is None or self.augmented_image.size == 0:
            QMessageBox.warning(self, "No Augmented Image", "There is no augmented image to save.")
            return
        self.save_augmented_image()

    def save_augmented_image(self):
        if not self.output_dir_set:
            self.prompt_for_output_dir()
        if self.augmented_image is not None:
            relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
            augmented_image_path = os.path.join(self.output_dir, relative_image_path)

            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)
            cv2.imwrite(augmented_image_path, self.augmented_image)

            relative_label_path = os.path.relpath(self.label_paths[self.current_image_path], self.dataset_root)
            augmented_label_path = os.path.join(self.output_dir, relative_label_path)

            os.makedirs(os.path.dirname(augmented_label_path), exist_ok=True)
            with open(augmented_label_path, 'w') as f:
                for polygon in self.augmented_polygons:
                    class_id = polygon[0]
                    coords = [f"{x / self.augmented_image_original_dims[1]} {y / self.augmented_image_original_dims[0]}" for x, y in polygon[1:]]
                    f.write(f"{class_id} {' '.join(coords)}\n")

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentDatasetGUI()
    ex.show()
    sys.exit(app.exec_())
