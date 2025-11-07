import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# Define the class names from the CIFAR-10 dataset
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
MODEL_FILENAME = 'cifar10_cnn_model.h5' 
CONFIDENCE_THRESHOLD = 0.40 

def load_and_preprocess_image(filepath):
    """Loads an image, resizes to 32x32, normalizes, and preps for model."""
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((32, 32), Image.LANCZOS)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_image(model, image_array):

    try:
        predictions = model.predict(image_array)
        
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        
        predicted_class = CLASS_NAMES[predicted_index]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

class ImageClassifierApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Image Classifier")
        self.root.geometry("700x560")
        
        self.title_label = Label(
            root, 
            text="CIFAR-10 Image Classifier",
            font=("Consolas", 16)
        )
        self.title_label.pack(pady=10)

        self.author_label = Label(
            root,
            text="By - Shrey Sharma",
            font=("Consolas", 12)
        )
        self.author_label.pack(pady=5)
        
        classes_text = "The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
        self.classes_label = Label(
            root,
            text=classes_text,
            font=("Consolas", 10),
            wraplength=1100 
        )
        self.classes_label.pack(pady=10)
        
        self.image_frame = Frame(
            root, 
            bd=1,
            relief=tk.SUNKEN,
            width=250,
            height=250
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = Label(self.image_frame)
        self.image_label.pack(expand=True)

        self.prediction_label = Label(
            root,
            text="Prediction: ---",
            font=("Consolas", 14)
        )
        self.prediction_label.pack(pady=10)
        
        self.status_label = Label(
            root,
            text="Please open an image file"
        )
        self.status_label.pack(pady=5)

        self.open_button = Button(
            root,
            text="Upload Image", 
            command=self.open_and_predict
        )
        self.open_button.pack(pady=10)

    def open_and_predict(self):
        try:
            filepath = filedialog.askopenfilename(
                title="Select an Image",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
            )
            if not filepath:
                return 

            self.status_label.config(text=os.path.basename(filepath)) 
            img_for_display = Image.open(filepath)
            img_for_display.thumbnail((240, 240))
            img_tk = ImageTk.PhotoImage(img_for_display)
            
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            image_array = load_and_preprocess_image(filepath)
            
            if image_array is None:
                self.prediction_label.config(text="Error: Could not process image")
                return

            predicted_class, confidence = predict_image(self.model, image_array)


            if predicted_class == "Error":
                self.prediction_label.config(text="Error during prediction",)
            elif confidence < CONFIDENCE_THRESHOLD:
                
                self.prediction_label.config(
                    text="This image does not belong to any of the 10 categories.",
                    fg="red"
                )
            else:

                self.prediction_label.config(
                    text=f"Prediction:{predicted_class} Confidence:{confidence*100:.2f}%",
                    fg="green"
                )
            

        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            self.prediction_label.config(text="Prediction: ---")

if __name__ == "__main__":
    
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        print(f"Successfully loaded model from {MODEL_FILENAME}")
        
        app_root = tk.Tk()
        app = ImageClassifierApp(app_root, model)
        app_root.mainloop()
        
    except IOError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

