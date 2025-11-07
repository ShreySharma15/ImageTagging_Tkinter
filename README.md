# Image Tagging AI (Tkinter Version)

**Image Tagging AI (Tkinter Version)** is a **desktop-based deep learning application** built using **TensorFlow** and **Tkinter**.  
It allows users to **upload an image** and get **real-time predictions** of what the image contains — powered by a trained **Convolutional Neural Network (CNN)**.

This is the **offline GUI version** of the project, designed for users who prefer a simple, local interface instead of a web-based Streamlit app.

---

## Features
- **Interactive Tkinter GUI** for selecting and tagging images  
- **CNN model** trained using **TensorFlow/Keras**  
- **Instant prediction results** on uploaded images  
- **Offline application** (no need for Streamlit or hosting)  
- Clean and minimal interface  

---

## Model Details
The model is a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset**, which contains:
- 60,000 images (32×32 pixels)
- 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

The CNN includes:
- Convolutional + MaxPooling layers  
- Dropout regularization  
- Dense layers with ReLU activation  
- Softmax output for multi-class classification  

---

## Project Structure
ImageTaggingAI-Tkinter/
│
├── image_tagging_app.py # Main Tkinter app file
├── model.h5 # Trained CNN model
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── sample_images/ # Example test images

---

## Installation & Setup (Run Locally)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShreySharma15/ImageTaggingAI-Tkinter.git
   cd ImageTaggingAI-Tkinter

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt

3. **Run the application**
    ```bash
    python image_tagging_app.py


---

## Requirements
- Python 3.8 or above
- TensorFlow
- Tkinter (pre-installed with Python)
- NumPy
- Pillow (for image handling)

---

## 👨‍💻 Author

**Shrey Sharma**  
SRM University, KTR Campus  

AI/ML Enthusiast | Developer | Innovator  

GitHub: [@ShreySharma15](https://github.com/ShreySharma15)
