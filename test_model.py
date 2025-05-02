import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_data

# Load model
model_path = "logo_recognition_model.h5"
if not os.path.exists(model_path):
    print("Error: Trained model not found. Run train_model.py first.")
    exit()

model = load_model(model_path)

# Load class names from the test dataset
data_dir = "dataset/test"
annotation_file = os.path.join(data_dir, "_annotations.csv")
_, _, class_names = load_data(data_dir, annotation_file)


# Function to predict the logo in an image
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read {image_path}")
        return

    img_resized = cv2.resize(img, (224, 224)) / 255.0  # Resize and normalize
    prediction = model.predict(np.expand_dims(img_resized, axis=0))
    label = class_names[np.argmax(prediction)]
    print(f"Predicted Label: {label}")


# Test on a single image
test_image = "dataset/Logos.v2-batch_alel.tensorflow/test/charlotte_1170435061632041947_20160125_jpg.rf.060aee4a223c2c83d07d73a4ecebdd9e.jpg"
predict_image(test_image)
