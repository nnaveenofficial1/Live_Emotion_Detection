import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 1. Load the trained model
print("Loading model...")
model_path = 'live_emotion_detection_model.keras'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit(1)

model = load_model(model_path)

# Define emotion labels (alphabetical order matches ImageDataGenerator behavior)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. Use Haarcascade to detect faces
# OpenCV provides a pre-trained Haarcascade XML file for frontal face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error: Failed to load cascade classifier from '{cascade_path}'")
    exit(1)

# 2. Use OpenCV to access webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Started live emotion detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert full frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # 4. For each detected face:
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) of the face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to (48, 48) as expected by the model
        roi_resized = cv2.resize(roi_gray, (48, 48))
        
        # Normalize (divide by 255)
        roi_normalized = roi_resized / 255.0
        
        # Reshape to match the input shape of the model (batch_size, height, width, channels)
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
        
        # Predict emotion using the model
        prediction = model.predict(roi_reshaped, verbose=0)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]
        
        # 5. Display bounding box and predicted emotion label on screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue bounding box
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show live video output
    cv2.imshow('Live Emotion Detection', frame)

    # Press 'q' to quit the live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
