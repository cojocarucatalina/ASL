import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import time
import os
from difflib import get_close_matches

clf = joblib.load('letter/asl_model.pkl')

feature_names = [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class ASLWordPredictor:
    def __init__(self):
        self.letter_buffer = []
        self.word_buffer = []
        self.last_letter = None
        self.letter_time = time.time()
        self.stable_time = 5.0
        self.space_gesture_time = 5.0
        self.last_detection_time = time.time()
        self.dictionary = self.load_dictionary()
        self.suggestions = []
        self.current_suggestion_index = 0
        self.suggestion_mode = False

    def load_dictionary(self):
        """Load a dictionary of common English words"""
        try:
            # Try to use a built-in word list if available
            word_file = "./word/dictionary"
            if os.path.exists(word_file):
                with open(word_file, 'r') as f:
                    return set(word.strip().lower() for word in f)
            else:
                return {"error"}
        except:
            return {"the", "and", "you", "hello", "sign", "language"}

    def add_letter(self, letter):
        current_time = time.time()

        if letter != self.last_letter:
            self.last_letter = letter
            self.letter_time = current_time
            return

        if current_time - self.letter_time > self.stable_time and letter not in self.letter_buffer[-1:]:
            self.letter_buffer.append(letter)
            print(f"Letter added: {letter}")
            self.update_suggestions()

        self.last_detection_time = current_time

    def check_for_space(self):
        current_time = time.time()

        if current_time - self.last_detection_time > self.space_gesture_time and self.letter_buffer:
            self.complete_word()
            self.word_buffer.append(" ")
            self.suggestions = []
            print("Space added")

    def complete_word(self):
        if self.letter_buffer:
            word = ''.join(self.letter_buffer)

            # Check if we should use a suggested correction
            if self.suggestion_mode and self.suggestions:
                word = self.suggestions[self.current_suggestion_index]

            self.word_buffer.append(word)
            self.letter_buffer = []
            self.suggestions = []
            self.suggestion_mode = False
            print(f"Word completed: {word}")

    def update_suggestions(self):
        """Update word suggestions based on current letter buffer"""
        if not self.letter_buffer:
            self.suggestions = []
            return

        current_word = ''.join(self.letter_buffer).lower()

        predictions = [word for word in self.dictionary if word.startswith(current_word)][:5]

        corrections = get_close_matches(current_word, self.dictionary, n=3, cutoff=0.6)

        combined = []
        for word in predictions + corrections:
            if word not in combined:
                combined.append(word)

        self.suggestions = combined[:5]  # Limit to top 5 suggestions
        self.current_suggestion_index = 0

    def cycle_suggestions(self):
        """Cycle through the available suggestions"""
        if self.suggestions:
            self.current_suggestion_index = (self.current_suggestion_index + 1) % len(self.suggestions)

    def toggle_suggestion_mode(self):
        """Toggle whether to use suggestions"""
        if self.suggestions:
            self.suggestion_mode = not self.suggestion_mode

    def backspace(self):
        """Remove the last letter"""
        if self.letter_buffer:
            self.letter_buffer.pop()
            self.update_suggestions()

    def get_current_text(self):
        """Get current text (words + current spelling)"""
        current_spelling = ''.join(self.letter_buffer)
        text = ''.join(self.word_buffer) + current_spelling
        return text

    def clear_text(self):
        """Clear all text"""
        self.letter_buffer = []
        self.word_buffer = []
        self.last_letter = None
        self.suggestions = []
        self.suggestion_mode = False


def is_pinch_gesture(landmarks_df):
    """Detect a pinch gesture (thumb and index finger touching)"""
    # This is a simplified implementation - you might need to adapt it
    try:
        # Extract thumb tip and index finger tip coordinates
        thumb_tip_x = landmarks_df.iloc[0, landmarks_df.columns.get_loc('landmark_4_x')]
        thumb_tip_y = landmarks_df.iloc[0, landmarks_df.columns.get_loc('landmark_4_y')]
        index_tip_x = landmarks_df.iloc[0, landmarks_df.columns.get_loc('landmark_8_x')]
        index_tip_y = landmarks_df.iloc[0, landmarks_df.columns.get_loc('landmark_8_y')]

        # Calculate Euclidean distance
        distance = np.sqrt((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)

        # If the distance is less than a threshold, it's a pinch
        return distance < 0.05
    except:
        return False


def is_flat_hand_gesture(landmarks_df):
    """Detect a flat hand gesture (all fingers extended)"""
    try:
        for finger in range(1, 5):
            tip_y = landmarks_df.iloc[0, landmarks_df.columns.get_loc(f'landmark_{4 * finger + 4}_y')]
            mcp_y = landmarks_df.iloc[0, landmarks_df.columns.get_loc(f'landmark_{4 * finger + 1}_y')]
            if tip_y > mcp_y:
                return False
        return True
    except:
        return False

word_predictor = ASLWordPredictor()

cap = cv2.VideoCapture(0)

ui_font = cv2.FONT_HERSHEY_SIMPLEX
ui_color = (0, 255, 0)
ui_thickness = 2

last_pinch_time = 0
last_flat_hand_time = 0
gesture_cooldown = 1.0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_detected = False

        current_time = time.time()

        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks_df = pd.DataFrame([landmarks], columns=feature_names)

                letter = clf.predict(landmarks_df)[0]

                word_predictor.add_letter(letter)

                if is_pinch_gesture(landmarks_df) and current_time - last_pinch_time > gesture_cooldown:
                    word_predictor.toggle_suggestion_mode()
                    last_pinch_time = current_time

                if is_flat_hand_gesture(landmarks_df) and current_time - last_flat_hand_time > gesture_cooldown:
                    word_predictor.cycle_suggestions()
                    last_flat_hand_time = current_time

                cv2.putText(frame, f"Letter: {letter}", (50, 50), ui_font, 1, ui_color, ui_thickness)

        else:
            word_predictor.check_for_space()

        current_text = word_predictor.get_current_text()

        letter_text = ''.join(word_predictor.letter_buffer)
        cv2.putText(frame, f"Spelling: {letter_text}", (50, 100), ui_font, 1, ui_color, ui_thickness)

        if word_predictor.suggestions:
            for i, suggestion in enumerate(word_predictor.suggestions):
                color = (
                0, 255, 255) if i == word_predictor.current_suggestion_index and word_predictor.suggestion_mode else (
                200, 200, 200)
                cv2.putText(frame, suggestion, (frame.shape[1] - 200, 50 + i * 30), ui_font, 0.7, color, ui_thickness)

        if word_predictor.suggestion_mode and word_predictor.suggestions:
            cv2.putText(frame, "Using suggestions (ON)", (50, 150), ui_font, 0.7, (0, 255, 255), 1)
        elif word_predictor.suggestions:
            cv2.putText(frame, "Using suggestions (OFF)", (50, 150), ui_font, 0.7, (200, 200, 200), 1)

        max_text_width = 40
        text_lines = [current_text[i:i + max_text_width] for i in range(0, len(current_text), max_text_width)]
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (50, 200 + i * 50), ui_font, 1, ui_color, ui_thickness)

        cv2.putText(frame, "Hold a letter to add it", (50, frame.shape[0] - 150), ui_font, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "No hand for 2s = space", (50, frame.shape[0] - 120), ui_font, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Pinch gesture = toggle suggestion", (50, frame.shape[0] - 90), ui_font, 0.7,
                    (255, 255, 255), 1)
        cv2.putText(frame, "Flat hand = cycle suggestions", (50, frame.shape[0] - 60), ui_font, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Press C = clear text, B = backspace, SPACE = complete word, Q = quit",
                    (50, frame.shape[0] - 30), ui_font, 0.7, (255, 255, 255), 1)

        cv2.imshow("ASL Recognition with Word Prediction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word_predictor.clear_text()
        elif key == ord('b'):
            word_predictor.backspace()
        elif key == 32:
            word_predictor.complete_word()
            word_predictor.word_buffer.append(" ")

cap.release()
cv2.destroyAllWindows()