import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained ASL model
MODEL_PATH = "random_forest.pkl"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("asl.mp4")

# Get original video width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Desired output window size
output_width = 640
output_height = 480

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks).reshape(1, -1)
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit within the desired window size
    frame_resized = cv2.resize(frame, (output_width, output_height))

    landmarks = extract_landmarks(frame_resized)
    if landmarks is not None:
        prediction = model.predict(landmarks)
        predicted_letter = prediction[0]

        cv2.putText(frame_resized, f"Letter: {predicted_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

