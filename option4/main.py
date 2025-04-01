import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def load_model(model_path):
    """Load a trained model from a pickle file"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['model'], model_data['scaler'], model_data['classes']


def extract_features(frame, holistic):
    """Extract features from a video frame using MediaPipe"""
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = holistic.process(frame_rgb)

    # Extract features (similar to feature extraction file)
    features = []

    # Example: Face landmarks
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

    # Example: Pose landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

    # Draw landmarks on the frame for visualization
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # If no landmarks were detected, return None
    if not features:
        return None, frame

    return np.array(features), frame


def classify_video(video_path, model_path):
    """Classify frames in a video using the specified model"""
    # Load model and scaler
    model, scaler, classes = load_model(model_path)
    model_name = os.path.basename(model_path).split('.')[0]

    # Initialize video capture
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # For smoothing predictions
    prediction_history = deque(maxlen=5)

    # Initialize counters
    frame_count = 0
    fps_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features
        features, annotated_frame = extract_features(frame, holistic)

        # If features were extracted, make a prediction
        if features is not None:
            # Handle variable-length feature vectors
            # This is a simplified approach - you might need to adapt based on your model's input dimensions
            expected_length = model.n_features_in_ if hasattr(model, 'n_features_in_') else None

            if expected_length is not None:
                if len(features) < expected_length:
                    features = np.pad(features, (0, expected_length - len(features)))
                elif len(features) > expected_length:
                    features = features[:expected_length]

            # Scale features
            features_scaled = scaler.transform([features])

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_history.append(prediction)

            # Get most common prediction from history for smoothing
            if prediction_history:
                from collections import Counter
                most_common = Counter(prediction_history).most_common(1)
                smoothed_prediction = most_common[0][0]

                # Convert numeric class to character (a-z)
                predicted_class = chr(ord('a') + smoothed_prediction)

                # Display prediction on frame
                cv2.putText(annotated_frame,
                            f"Class: {predicted_class}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1.0:  # update FPS every second
            fps_display = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()

        cv2.putText(annotated_frame,
                    f"FPS:",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Display model name
        cv2.putText(annotated_frame,
                    f"Model: {model_name}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Display the frame
        cv2.imshow('Video Classification', annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Classify video frames using trained models')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--model', type=str, default='nn.pkl',
                        choices=['nn.pkl', 'forest.pkl', 'gradient_boost.pkl'],
                        help='Model to use for classification')

    args = parser.parse_args()

    print(f"Using model: {args.model} to classify video: {args.video}")
    classify_video(args.video, args.model)