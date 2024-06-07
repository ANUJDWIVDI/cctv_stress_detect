import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Load your models here
# Using MobileNetV2 for expression analysis as an example
expression_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3), pooling='avg')
# For sharpness detection, we will use a simple Laplacian variance method, so no need to load a model

def process_frame(frame):
    # Example face detection and expression analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if measure_sharpness(face) < 100:
            face = enhance_sharpness(face)
        expression = analyze_expression(face)
        # Draw bounding box and expression on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def detect_faces(gray):
    # Use OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def measure_sharpness(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def enhance_sharpness(face):
    # Using unsharp mask as an example
    gaussian = cv2.GaussianBlur(face, (9, 9), 10.0)
    sharp_face = cv2.addWeighted(face, 1.5, gaussian, -0.5, 0)
    return sharp_face

def analyze_expression(face):
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    expression = expression_model.predict(face)
    # For simplicity, returning a dummy expression. You should map the output to actual expressions.
    return "Neutral" if expression[0][0] < 0.5 else "Happy"

# Example code to read a video frame and process it
cap = cv2.VideoCapture(0)  # Open the default camera

frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    # Save the frame to a file instead of displaying it
    cv2.imwrite(f'output_frame_{frame_counter}.jpg', processed_frame)
    frame_counter += 1

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
