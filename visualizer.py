import tkinter as tk
from tkinter import ttk, Canvas
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

class PolygonVisualizerApp:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.root = tk.Tk()
        self.root.title("Polygon Visualizer")

        self.root.geometry("1280x720")

        self.current_video_name = tk.StringVar()
        self.current_split = "train"
        self.frames = {'train': [], 'val': []}
        self.current_frame_index = 0

        self.video_names = self.get_video_names()
        self.setup_video_selection()
        self.setup_image_slider() 
        self.setup_tabs()
        self.setup_navigation_controls()

    # Class ID to color mapping
    class_colors = {
        '0': '#1F77B4',
        '1': '#AEC7E8',
        '2': '#FF7F0E',
        '3': '#2CA02C',
        '4': '#98DF8A',
        '5': '#FF9896',
        '6': '#9467BD',
        '7': '#8C564B',
        '8': '#C49C94',
        '9': '#E377C2',
        '10': '#7F7F7F',
        '11': '#C7C7C7',  # You might not need this if you don't have a class 11
        '12': '#DBDB8D',  # You might not need this if you don't have a class 12
        '13': '#17BECF',  # You might not need this if you don't have a class 13
        '14': '#9EDAE5',  # You might not need this if you don't have a class 14
    }

    class_names = {
        '0': 'buoy',
        '1': 'buoy_glyph_1',
        '2': 'buoy_glyph_2',
        '3': 'buoy_glyph_3',
        '4': 'buoy_glyph_4',
        '5': 'gate',
        '6': 'earth_glyph',
        '7': 'torpedo_open',
        '8': 'torpedo_closed',
        '9': 'torpedo_hole',
        '10': 'bin',
        # '11', '12', '13', and '14' are placeholders in case you need to add more classes later.
    }

    def get_video_names(self):
        video_names = set()
        for split in ['train', 'val']:
            split_dir = os.path.join(self.base_dir, "images", split)
            if os.path.isdir(split_dir):
                for name in os.listdir(split_dir):
                    if os.path.isdir(os.path.join(split_dir, name)):
                        video_names.add(name)
        return sorted(video_names)

    def setup_video_selection(self):
        selection_frame = tk.Frame(self.root)
        selection_frame.pack(fill="x")
        tk.Label(selection_frame, text="Select Video:").pack(side="left", padx=5, pady=5)
        video_selector = ttk.Combobox(selection_frame, textvariable=self.current_video_name, values=self.video_names, state="readonly")
        video_selector.pack(side="left", fill="x", expand=True)
        video_selector.bind("<<ComboboxSelected>>", self.load_video_frames)

    def setup_image_slider(self):
            # This frame will contain the slider
            slider_frame = tk.Frame(self.root)
            slider_frame.pack(fill='x', padx=5, pady=5)

            # Initialize the slider (scale) widget
            self.image_slider = tk.Scale(slider_frame, from_=0, to=100, orient='horizontal', command=self.on_slider_change)
            self.image_slider.pack(fill='x', expand=True)

    def setup_tabs(self):
        self.tab_control = ttk.Notebook(self.root)
        self.tabs = {}
        for split in ['train', 'val']:
            tab = ttk.Frame(self.tab_control)
            self.tab_control.add(tab, text=split.capitalize())
            canvas = Canvas(tab)
            canvas.pack(expand=True, fill="both")
            self.tabs[split] = {'canvas': canvas, 'images': []}
        self.tab_control.pack(expand=1, fill="both")
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def setup_navigation_controls(self):
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(fill="x", side="bottom")
        prev_button = tk.Button(buttons_frame, text="<< Previous", command=lambda: self.navigate(-1))
        prev_button.pack(side="left")
        next_button = tk.Button(buttons_frame, text="Next >>", command=lambda: self.navigate(1))
        next_button.pack(side="right")

    def load_video_frames(self, event=None):
        self.frames = {'train': [], 'val': []}
        video_name = self.current_video_name.get()
        for split in ['train', 'val']:
            img_dir = os.path.join(self.base_dir, "images", split, video_name)
            label_dir = os.path.join(self.base_dir, "labels", split, video_name)
            self.frames[split] = []
            if os.path.isdir(img_dir) and os.path.isdir(label_dir):
                for frame in sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0])):
                    img_path = os.path.join(img_dir, frame)
                    label_path = os.path.join(label_dir, os.path.splitext(frame)[0] + ".txt")
                    if os.path.isfile(img_path) and os.path.isfile(label_path):
                        self.frames[split].append((img_path, label_path))
        self.display_image(self.current_split, 0)

        # Update the slider's range based on the number of frames in the current split
        if self.frames[self.current_split]:
            self.image_slider.config(from_=0, to=len(self.frames[self.current_split]) - 1)
        else:
            self.image_slider.config(from_=0, to=0)

    def on_tab_change(self, event):
        self.current_split = self.tab_control.tab(self.tab_control.select(), "text").lower()
        self.display_image(self.current_split, 0)

        # Update the slider's range when the tab changes
        self.load_video_frames()

    def navigate(self, direction):
        self.current_frame_index += direction
        self.current_frame_index = max(0, min(self.current_frame_index, len(self.frames[self.current_split]) - 1))
        self.display_image(self.current_split, self.current_frame_index)

        # Update the slider position when navigating
        self.image_slider.set(self.current_frame_index)

    def on_slider_change(self, event):
            # Change the displayed image when the slider is adjusted
            self.current_frame_index = self.image_slider.get()
            self.display_image(self.current_split, self.current_frame_index)
            
    def draw_text_with_outline(self, draw, position, text, font, fill_color="white", outline_color="black"):
        # Draw the outline by offsetting the position in all directions
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for offset in offsets:
            offset_position = (position[0] + offset[0], position[1] + offset[1])
            draw.text(offset_position, text, font=font, fill=outline_color)

        # Draw the main text
        draw.text(position, text, font=font, fill=fill_color)
        
    def draw_text_with_outline(self, draw, position, text, font, fill_color="white", outline_color="black"):
        # Draw the outline by offsetting the position in all directions
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for offset in offsets:
            offset_position = (position[0] + offset[0], position[1] + offset[1])
            draw.text(offset_position, text, font=font, fill=outline_color)

        # Draw the main text
        draw.text(position, text, font=font, fill=fill_color)

    def display_image(self, split, index):
        if not self.frames[split]:
            return

        img_path, label_path = self.frames[split][index]
        image = Image.open(img_path)

        # Ensure the main image is in RGBA mode to support transparency
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Resize the image first to fit the canvas
        canvas = self.tabs[split]['canvas']
        canvas.update_idletasks()  # Ensure canvas size is updated
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        resized_image = image.resize((canvas_width, canvas_height), Image.ANTIALIAS)

        # Temporary image for drawing semi-transparent polygons
        temp_image = Image.new('RGBA', resized_image.size, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_image)

        font = ImageFont.truetype("arial.ttf", 24)  # Adjust the font size and path as needed

        for line in open(label_path, 'r'):
            parts = line.strip().split()
            class_id = parts[0]
            color = self.class_colors.get(class_id, 'yellow')  # Use class-specific color

            # Convert hex color to RGBA with semi-transparency for the fill
            fill_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (128,)

            polygon = [(float(parts[i]) * canvas_width, float(parts[i + 1]) * canvas_height) for i in range(1, len(parts), 2)]

            # Draw the polygon with a thicker line and semi-transparent fill on the temporary image
            temp_draw.polygon(polygon, outline=color, fill=fill_color, width=3)

        # Composite the temporary image with the main image
        resized_image.alpha_composite(temp_image)

        # Now, draw points and text on the composited image
        final_draw = ImageDraw.Draw(resized_image)
        for line in open(label_path, 'r'):
            parts = line.strip().split()
            polygon = [(float(parts[i]) * canvas_width, float(parts[i + 1]) * canvas_height) for i in range(1, len(parts), 2)]
            
            # Draw points on each vertex
            for (x, y) in polygon:
                r = 2  # Radius for the points
                final_draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="red")

            class_name = self.class_names.get(parts[0], 'Unknown')
            # Adjust text position to avoid overlap with polygon edges
            text_position = (polygon[0][0] + 10, polygon[0][1] + 10)

            # Draw the class name with a black border and white fill
            self.draw_text_with_outline(final_draw, text_position, class_name, font)

        # Display the modified image on the canvas
        canvas_image = ImageTk.PhotoImage(resized_image)
        canvas.create_image(0, 0, anchor="nw", image=canvas_image)
        canvas.image = canvas_image  # Keep a reference


    def run(self):
        self.root.mainloop()

# Usage example
base_dir = "C:\\Users\\Chef\\Desktop\\HACKERMAN\\Programming\\Python Projects\\yoloTrainer\\TrainingData"
app = PolygonVisualizerApp(base_dir)
app.run()

