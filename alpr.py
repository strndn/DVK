import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox
from PIL import Image, ImageTk

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import requests
import torch
import re

from PIL import Image

from ultralytics import YOLO
import cv2

import os
import shutil

# Load the OCR processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
# Initialize the YOLO model
model = YOLO("YOUR-YOLO-MODEL")
# Global variable to hold the file path of the uploaded image
file_path = ''
prev_file_path = ''
boxes = []

class DragDropImageApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Configure the main window
        self.title("License Plate Recognizer")
        self.geometry("1500x1000")

        # Create a canvas to display the image
        self.canvas = tk.Canvas(self, width=1400, height=900, highlightthickness=1, highlightbackground="gray")
        self.canvas.pack()

        # Create a label for the initial message in the drop area
        self.drop_message = tk.Label(self.canvas, text="Drop an image here", font=("Arial", 16), fg="gray")
        self.drop_message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Register the window as a file drop target
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)

        # Create a label to display recognized text below the canvas
        self.text_label = tk.Label(self, text="")
        self.text_label.pack(pady=5)

        # Create a frame to hold the button and radio buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        # Create a button below the text label
        self.button = tk.Button(self.button_frame, text="Submit", command=self.on_button_click)
        self.button.pack(side=tk.LEFT, padx=10)

        # Variable to hold the selected radio button value
        self.selected_option = tk.IntVar()
        self.selected_option.set(12)  # Set the default value for the radio buttons to 12 layers

        # Add a trace method to call `update_selected_option` when `selected_option` changes
        self.selected_option.trace_add('write', self.update_selected_option)

        # Create 5 radio buttons for layer options
        options = [4, 6, 8, 12, 16]
        labels = ["4 layers", "6 layers", "8 layers", "12 layers", "16 layers"]
        for i, option in enumerate(options):
            rb = tk.Radiobutton(self.button_frame, text=labels[i], variable=self.selected_option, value=option)
            rb.pack(side=tk.LEFT)

        # Initialize the model
        self.trained_model = None
        self.update_selected_option()  # Initialize the trained model based on the default selected option

        # Bind the closing event to the clean-up function
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_selected_option(self, *args):
        # Update the model and/or other parameters based on the selected radio button
        selected_option = self.selected_option.get()

        # Perform different actions based on the selected option
        if selected_option == 4:
            # Action for 4 layers
            self.update_model("PATH-TO-MODEL-WITH-4-LAYERS")
        elif selected_option == 6:
            # Action for 6 layers
            self.update_model("PATH-TO-MODEL-WITH-6-LAYERS")
        elif selected_option == 8:
            # Action for 8 layers
            self.update_model("PATH-TO-MODEL-WITH-8-LAYERS")
        elif selected_option == 12:
            # Action for 12 layers
            self.update_model("PATH-TO-MODEL-WITH-12-LAYERS")
        elif selected_option == 16:
            # Action for 16 layers
            self.update_model("PATH-TO-MODEL-WITH-16-LAYERS")

    def update_model(self, model_name):
        # Load the model based on the given model name
        self.trained_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Model updated to {model_name}")

    def on_closing(self):
        # Clear the cropped images folder before closing the application
        self.clear_folder("PATH-TO-CROPPED-IMAGES-FOLDER")

        # Clear the edited image folder before closing the application
        self.clear_folder("PATH-TO-EDITED-IMAGE-FOLDER")
        
        # Call the destroy method to close the application window
        self.destroy()

    def clear_folder(self, folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all files in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # Remove the file
                os.remove(file_path)

    def on_drop(self, event):
        global file_path
        # Handle the file drop event
        file_path = event.data.strip()
        print("File dropped:", file_path)

        # Check if the file is an image
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Display the image on the canvas
            self.show_image(file_path)
        else:
            messagebox.showerror("Error", "File is not an image")

    def show_image(self, file_path):
        # Open the image using Pillow
        img = Image.open(file_path)

        # Get the dimensions of the image and the canvas
        img_width, img_height = img.size
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()

        # Calculate the scale factor to fit the image inside the canvas while maintaining aspect ratio
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        scale_factor = min(scale_width, scale_height)

        # Calculate the new width and height based on the scale factor
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Resize the image while maintaining the aspect ratio
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert the image to a format tkinter can use
        img_tk = ImageTk.PhotoImage(img)

        # Clear the canvas and display the new image
        self.canvas.delete("all")

        # Remove the drop message label when an image is displayed
        self.drop_message.place_forget()

        # Calculate the position to center the image in the canvas
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2

        # Create the image on the canvas at the calculated position
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)
        # Keep a reference to the image
        self.canvas.image = img_tk

        # Set the border to disappear after the image is displayed
        self.canvas.configure(highlightthickness=0)

    def on_button_click(self):
        self.text_label.config(text="")

        global file_path
        lp_id_text = ''
        i = 0
        if file_path == '':
            messagebox.showerror("Error", "No image uploaded")
        else:
            # Get the selected option from the radio buttons
            selected_option = self.selected_option.get()

            list_of_cropped_images = self.crop_image(selected_option)
            for cropped_image in list_of_cropped_images:
                lp_id = self.generate_lp_id(cropped_image)
                if len(lp_id) >= 5:
                    i += 1
                    lp_id_text += f"{i}: {lp_id} "
                    self.text_label.config(text=lp_id_text, font=("Arial", 16))
                    self.text_label.pack()
                    print(f"License Plate ID: {lp_id}")

    def crop_image(self, selected_option):
        # Utilize the selected option in the cropping process
        # Add your own logic here to utilize the selected option value

        global boxes, file_path, prev_file_path

        if (file_path != prev_file_path):
            # Run the YOLO model with a confidence threshold
            results = model(file_path, conf=0.5)
            prev_file_path = file_path
            # Get bounding boxes from the model results
            boxes = results[0].boxes.xyxy.tolist()

        # Load the input image using OpenCV
        img = cv2.imread(file_path)
        
        # Create a list to hold cropped image file paths
        list_of_cropped_images = []

        # Loop through the detected bounding boxes
        for i, box in enumerate(boxes):
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box
            
            # Draw a red rectangle around the bounding box on the input image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 7)
            
            # Crop the object from the image using the bounding box coordinates
            crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            
            # Save the cropped image
            cropped_image_path = f"PATH-TO-CROPPED-IMAGES/{i}.jpg"
            cv2.imwrite(cropped_image_path, crop_object)
            list_of_cropped_images.append(cropped_image_path)
            
            # Generate text for the cropped image using OCR
            lp_id = self.generate_lp_id(cropped_image_path)
            if (len(lp_id) < 5):
                lp_id = f"Invalid LP generated: {lp_id}"
            
            # Start with an initial font scale
            font_scale = 3.0

            # Write the generated text on top of the bounding box on the input image
            # Calculate text position
            text_position = (int(x1), int(y1) - 20)  # Place text above the bounding box
            text_size = cv2.getTextSize(lp_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 7)[0]

            # Calculate the background rectangle
            rect_start = (text_position[0] - 10, text_position[1] - text_size[1] - 10)
            rect_end = (text_position[0] + text_size[0] + 10, text_position[1] + 10)
        
            # Draw the white rectangle as the background
            cv2.rectangle(img, rect_start, rect_end, (255, 255, 255), -1)

            cv2.putText(img, lp_id, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 7)
        
        # Save the modified image with bounding boxes and text
        edited_image_path = "PATH-TO-EDITED-IMAGE/edited_image.jpg"
        cv2.imwrite(edited_image_path, img)
        
        # Show the edited image on the canvas
        self.show_image(edited_image_path)
        
        # Return the list of cropped images for further processing if needed
        return list_of_cropped_images

    def generate_lp_id(self, file_path):
        img = Image.open(file_path).convert('RGB')
        pixel_values = processor(img, return_tensors='pt').pixel_values.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        generated_ids = self.trained_model.generate(pixel_values)
        generated_lp = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        regex_generated_lp = re.sub(r'[^A-Za-z0-9]', '', generated_lp)
        return regex_generated_lp

# Create the application and run it
if __name__ == "__main__":
    app = DragDropImageApp()
    app.mainloop()