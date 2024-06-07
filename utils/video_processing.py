import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

expression_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3), pooling='avg')


def process_video(input_filepath, output_filepath):
    # Open the input video file
    cap = cv2.VideoCapture(input_filepath)

    # Initialize video writer to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    # Process each frame and write it to the output video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)  # Assuming you have a function to process each frame
        out.write(processed_frame)

    # Release video writer and capture objects
    out.release()
    cap.release()


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    emotion_data = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = analyze_expression(face)
        print("Emotion detected:", emotion)
        emotion_data.append((x, y, w, h, emotion))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame, emotion_data

def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("Number of faces detected:", len(faces))
    return faces

def analyze_expression(face):
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    expression = expression_model.predict(face)
    print("Expression probabilities:", expression)
    return "Neutral" if expression[0][0] < 0.5 else "Happy"
