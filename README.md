# LOGO_RECOGNITION

🔍 Real-Time Logo Recognition Using MobileNetV2
This project is a real-time logo recognition system built with Python, OpenCV, and TensorFlow (MobileNetV2). It captures video through a webcam and identifies brand logos using a trained deep learning model. The application is designed for high accuracy and quick response in dynamic environments.

🚀 Features
Real-time logo detection using webcam input

Deep learning model trained on the Logo-2K+ dataset (2,341 logo classes)

Uses MobileNetV2 for fast and efficient classification

Displays identified brand names live on the video feed

Modular code with separate scripts for data loading, model training, testing, and recognition

🛠️ Project Structure
load_data.py – Prepares dataset using ImageDataGenerator and organizes image folders

train_model.py – Trains the MobileNetV2 model with logo images

test_model.py – Evaluates the trained model on test data

recognition.py – Launches webcam and performs real-time prediction

📁 Dataset
Logo-2K+: A large-scale logo dataset with over 160,000 images and 2,341 logo categories

Dataset is organized into subfolders by brand name for use with flow_from_directory

✅ Requirements
Python 3.x

TensorFlow

OpenCV

NumPy
