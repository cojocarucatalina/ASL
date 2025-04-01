# import cv2
# import mediapipe as mp
# import joblib
# import numpy as np
# import pandas as pd
# import tkinter as tk
# from tkinter import filedialog
# import matplotlib.pyplot as plt
#
# # Load trained model
# clf = joblib.load('letter/model.pkl')
# feature_names = [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# def process_frame(frame, hands):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)
#
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])
#
#             landmarks_df = pd.DataFrame([landmarks], columns=feature_names)
#
#             letter = clf.predict(landmarks_df)[0]
#             probabilities = clf.predict_proba(landmarks_df)[0]
#
#             # Sort probabilities in descending order
#             sorted_indices = np.argsort(probabilities)[::-1]
#             top_classes = clf.classes_[sorted_indices]
#             top_probs = probabilities[sorted_indices]
#
#             # Display prediction on image
#             cv2.putText(frame, f"Predicted Letter: {letter}", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # Create a bar plot of top predictions
#             plt.figure(figsize=(8, 4))
#             plt.bar(top_classes[:5], top_probs[:5])
#             plt.title('Top 5 Prediction Probabilities')
#             plt.xlabel('Letters')
#             plt.ylabel('Probability')
#             plt.ylim(0, 1)
#             plt.tight_layout()
#             plt.savefig('prediction_probabilities.png')
#             plt.close()
#
#             # Print detailed prediction information
#             print("\nPrediction Details:")
#             print(f"Predicted Letter: {letter}")
#             print("\nTop 5 Predictions:")
#             for cls, prob in zip(top_classes[:5], top_probs[:5]):
#                 print(f"{cls}: {prob:.4f}")
#
#     return frame
#
#
# def process_photo():
#     root = tk.Tk()
#     root.withdraw()
#
#     while True:
#         photo_path = filedialog.askopenfilename(title="Select an Image",
#                                                 filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
#         if not photo_path:
#             print("No file selected. Exiting.")
#             break
#
#         image = cv2.imread(photo_path)
#         if image is None:
#             print("Error: Could not read image.")
#             continue
#
#         with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#             processed_image = process_frame(image, hands)
#
#         # Show the processed image
#         cv2.imshow("ASL Letter Recognition - Photo", processed_image)
#
#         # Show the prediction probabilities plot
#         prob_image = cv2.imread('prediction_probabilities.png')
#         if prob_image is not None:
#             cv2.imshow("Prediction Probabilities", prob_image)
#
#         key = cv2.waitKey(0)
#
#         # Close the windows if 'q' is pressed or the close button is clicked
#         if key == ord('q'):
#             break
#
#         # Close both windows
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     process_photo()

import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Load trained model
clf = joblib.load('letter/model.pkl')
feature_names = [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def process_frame(frame, hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            landmarks_df = pd.DataFrame([landmarks], columns=feature_names)

            letter = clf.predict(landmarks_df)[0]
            probabilities = clf.predict_proba(landmarks_df)[0]

            # Sort probabilities in descending order
            sorted_indices = np.argsort(probabilities)[::-1]
            top_classes = clf.classes_[sorted_indices]
            top_probs = probabilities[sorted_indices]

            # Display prediction on image
            cv2.putText(frame, f"Predicted: {letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Create a bar plot of top predictions
            plt.figure(figsize=(8, 4))
            plt.bar(top_classes[:5], top_probs[:5])
            plt.title('Top 5 Prediction Probabilities')
            plt.xlabel('Letters')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('prediction_probabilities.png')
            plt.close()

            # Print detailed prediction information
            print("\nPrediction Details:")
            print(f"Predicted Letter: {letter}")
            for cls, prob in zip(top_classes[:5], top_probs[:5]):
                print(f"{cls}: {prob:.4f}")

    return frame


def process_photo():
    root = tk.Tk()
    root.withdraw()

    photo_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not photo_path:
        print("No file selected. Exiting.")
        return

    image = cv2.imread(photo_path)
    if image is None:
        print("Error: Could not read image.")
        return

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        processed_image = process_frame(image, hands)

    cv2.imshow("ASL Letter Recognition - Photo", processed_image)

    # Show the prediction probabilities plot
    prob_image = cv2.imread('prediction_probabilities.png')
    if prob_image is not None:
        cv2.imshow("Prediction Probabilities", prob_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            processed_frame = process_frame(frame, hands)
            cv2.imshow("ASL Letter Recognition - Webcam", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("Choose mode: (1) Photo or (2) Webcam: ")
    if choice == '1':
        process_photo()
    elif choice == '2':
        process_webcam()
    else:
        print("Invalid choice. Exiting.")
