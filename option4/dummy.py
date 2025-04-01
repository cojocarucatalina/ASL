import cv2
import mediapipe as mp
import argparse
import tkinter as tk
from tkinter import filedialog
import os

def select_image_from_folder():
    """Open a file dialog to select an image file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None


def detect_hand_landmarks(image_path=None, use_dialog=False):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    if use_dialog:
        image_path = select_image_from_folder()
        if not image_path:
            print("No image selected. Exiting.")
            return

    if image_path:
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find the image: {image_path}")
            return

        process_image(image, hands, mp_drawing, mp_drawing_styles, mp_hands)

        print("Press any key to exit...")
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(0)
        print("Using webcam. Press 'q' to exit.")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process frame
            process_image(image, hands, mp_drawing, mp_drawing_styles, mp_hands)

            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()


def process_image(image, hands, mp_drawing, mp_drawing_styles, mp_hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    annotated_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for i, landmark in enumerate(hand_landmarks.landmark):
                print(f"  Landmark {i}: ({landmark.x:.3f}, {landmark.y:.3f}, {landmark.z:.3f})")
    else:
        print("No hands detected in the image.")

    # Display the image with landmarks
    cv2.imshow("MediaPipe Hands", annotated_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect hand landmarks in an image using MediaPipe")
    parser.add_argument("--image", type=str, help="Path to the input image file (optional)")
    parser.add_argument("--dialog", action="store_true", help="Open a file dialog to select an image")
    args = parser.parse_args()

    detect_hand_landmarks(args.image, args.dialog)