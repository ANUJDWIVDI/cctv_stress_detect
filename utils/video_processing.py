import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from datetime import datetime

expression_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3), pooling='avg')
import shutil

def process_video(input_filepath, output_filepath):
    cap = cv2.VideoCapture(input_filepath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    # Create a list to store analysis results
    analysis_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, emotion_data = process_frame(frame)
        # Write the processed frame to the output video
        out.write(processed_frame)
        
        # Append emotion data to analysis results list
        analysis_results.extend(emotion_data)
    
    cap.release()
    out.release()
    
    # Save analysis results to a .txt file
    save_analysis_results(analysis_results, output_filepath + '.txt')
    
    # Copy the processed video to the desired location
    processed_video_path = 'uploads/processed_video.mp4'
    shutil.copy(output_filepath, processed_video_path)
    
    print("Video analysis completed.")



def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    emotion_data = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = analyze_expression(face)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        person = "Unknown"  # Placeholder for person identification
        state = "Normal"  # Placeholder for state information
        
        # Check if emotion is stressed, change box color to red
        if emotion == "Stressed":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        print("Emotion detected:", emotion)
        emotion_data.append((timestamp, person, emotion, state))
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
    
    # Define emotional thresholds
    stressed_threshold = 0.7
    relaxed_threshold = 0.2
    focused_threshold = 0.5
    distracted_threshold = 0.3
    confident_threshold = 0.6
    anxious_threshold = 0.8
    
    # Extract probabilities for each emotion
    stressed_prob = expression[0][0]
    relaxed_prob = expression[0][1]
    focused_prob = expression[0][2]
    distracted_prob = expression[0][3]
    confident_prob = expression[0][4]
    anxious_prob = expression[0][5]
    
    # Classify emotion based on probabilities
    if stressed_prob >= stressed_threshold:
        emotion = "Stressed"
    elif relaxed_prob >= relaxed_threshold:
        emotion = "Relaxed"
    elif focused_prob >= focused_threshold:
        emotion = "Focused"
    elif distracted_prob >= distracted_threshold:
        emotion = "Distracted"
    elif confident_prob >= confident_threshold:
        emotion = "Confident"
    elif anxious_prob >= anxious_threshold:
        emotion = "Anxious"
    else:
        emotion = "Unknown"
    
    print("Emotion probabilities:", expression)
    return emotion

def save_analysis_results(analysis_results, filepath):
    with open(filepath, 'w') as f:
        f.write("Timestamp\tPerson\tEmotion\tState\n")
        for result in analysis_results:
            f.write("\t".join(map(str, result)) + "\n")


