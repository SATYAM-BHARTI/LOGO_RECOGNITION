import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model_path = "logo_recognition_model.h5"
if not os.path.exists(model_path):
    print("Error: Trained model not found.")
    exit()

model = load_model(model_path)

# Load class names
class_names_path = "class_names.txt"
if not os.path.exists(class_names_path):
    print("Error: class_names.txt not found.")
    exit()

with open(class_names_path, "r") as f:
    class_names = [line.strip() for line in f]

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess
    img_resized = cv2.resize(frame, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)

    if predicted_index >= len(class_names):
        label = "Unknown"
    else:
        label = class_names[predicted_index]

    # Display
    cv2.putText(frame, f"Prediction: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Logo Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
