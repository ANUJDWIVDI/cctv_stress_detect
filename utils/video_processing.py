import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

expression_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3), pooling='avg')


import cv2

def process_video(input_filepath, output_filepath):
    cap = cv2.VideoCapture(input_filepath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame (replace this with your processing logic)
        processed_frame = frame
        
        # Write the processed frame to the output video
        out.write(processed_frame)
    
    cap.release()
    out.release()



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
