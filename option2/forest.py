import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATASET_PATH = "data"
DATA_SAVE_PATH = "landmarks_data.pkl"
MODEL_SAVE_PATH = "random_forest.pkl"


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

    return np.array(X), np.array(y)


def train_model():
    """Load dataset, train Random Forest model, and save it."""
    X, y = load_dataset()
    if len(X) == 0:
        print("No valid hand landmarks found! Check your dataset.")
        return

    with open(DATA_SAVE_PATH, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"Landmark data saved to {DATA_SAVE_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved as {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
