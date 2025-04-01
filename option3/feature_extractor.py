import cv2
import mediapipe as mp
import numpy as np
import os

from pandas.io import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATASET_PATH = "data"
DATA_SAVE_PATH = "landmarks_data.pkl"

def extract_hand_landmarks(image):
    """Extract hand landmarks using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
    return None


def load_dataset():
    """Extract landmarks from images and prepare dataset."""
    X, y = [], []
    for label in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue
        print(f"Processing label: {label}")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            landmarks = extract_hand_landmarks(image)
            if landmarks:
                X.append(landmarks)
                y.append(label)

    # Save extracted data
    with open(DATA_SAVE_PATH, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"Landmark data saved to {DATA_SAVE_PATH}")

    return np.array(X), np.array(y)