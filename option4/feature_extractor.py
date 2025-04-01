import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import csv
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands


def extract_hand_features(image_path):
    """Extract hand landmark features from an image using MediaPipe Hands"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image at {image_path}")
        return None

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3,
                        min_tracking_confidence=0.3) as hands:
        results = hands.process(image_rgb)
        features = []

        # Extract hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])

        # If no hand landmarks were detected, return an empty feature vector
        if not features:
            features = [0] * 63  # Placeholder for missing landmarks

        return np.array(features)


def extract_all_hand_features(data_dir='C:/Users/cojoc/Downloads/ASL_Alphabet_Dataset/asl_alphabet_train'):
    """Process all images in subdirectories and extract hand features, saving to CSV and Pickle"""
    features = []
    labels = []

    # Define CSV file name
    csv_filename = 'features_data.csv'

    # Iterate through subdirectories (a-z)
    for label in range(ord('B'), ord('D') + 1):
        label_char = chr(label)
        class_dir = os.path.join(data_dir, label_char)

        # Skip if directory doesn't exist
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {label_char}")
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in tqdm(image_files):
            image_path = os.path.join(class_dir, image_file)
            image_features = extract_hand_features(image_path)

            if image_features is not None:
                if len(features) > 0 and len(image_features) != len(features[0]):
                    if len(image_features) < len(features[0]):
                        image_features = np.pad(image_features, (0, len(features[0]) - len(image_features)))
                    else:
                        image_features = image_features[:len(features[0])]

                features.append(image_features)
                labels.append(ord(label_char) - ord('a'))  # Convert to numeric label

    X = np.array(features)
    y = np.array(labels)

    # Save as Pickle file
    # with open('features_data.pkl', 'a') as f:
    #     pickle.dump({'features': X, 'labels': y}, f)

    # Save as CSV file
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'feature_{i}' for i in range(X.shape[1])]
        writer.writerow(header)
        for label, feature_vector in zip(y, X):
            writer.writerow([label] + feature_vector.tolist())

    print(f"Extracted features from {len(features)} images")
    print(f"Feature vector shape: {X.shape}")
    print(f"Data saved to {csv_filename} and features_data.pkl")

    return X, y


if __name__ == "__main__":
    X, y = extract_all_hand_features()
    print("Feature extraction complete. Data saved.")
