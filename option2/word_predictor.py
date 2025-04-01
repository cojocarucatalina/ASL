import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

LETTER_MODEL_PATH = "svm_model.pkl"
with open(LETTER_MODEL_PATH, 'rb') as f:
    letter_model = pickle.load(f)

WORD_MODEL_PATH = "predictors.pkl"
with open(WORD_MODEL_PATH, 'rb') as f:
    word_model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/rac2.mp4")

# fps = cap.get(cv2.CAP_PROP_FPS)
#
# delay = int((1 / fps) * 1000 * 4)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     cv2.imshow("Slow Motion Video", frame)

############################

output_width, output_height = 640, 480

current_word = []
last_predicted_letter = None

def extract_landmarks(image):
    """Extract hand landmarks using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks).reshape(1, -1)
    return None

def predict_word_from_prefix(prefix):
    """Use the word predictor model to guess a word from the first few letters."""
    if len(prefix) == 0:
        return ""
    try:
        return word_model.predict([prefix])[0]
    except:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (output_width, output_height))
    landmarks = extract_landmarks(frame_resized)

    if landmarks is not None:
        predicted_letter = letter_model.predict(landmarks)[0]

        cv2.putText(frame_resized, f"Letter: {predicted_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_word_str = ''.join(current_word)

    predicted_word = predict_word_from_prefix(current_word_str)

    cv2.putText(frame_resized, f"Current Word: {current_word_str}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame_resized, f"Predicted Word: {predicted_word}", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("ASL Detection", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if predicted_letter and (not current_word or predicted_letter != current_word[-1]):
            current_word.append(predicted_letter)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
