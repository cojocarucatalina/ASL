# import cv2
# import mediapipe as mp
# import csv
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
#
# csv_file = open('asl_data.csv', 'a', newline='')  # 'a' to append, not overwrite
# csv_writer = csv.writer(csv_file)
# #csv_writer.writerow(['letter'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']])
#
# x_mark_x = 20
# x_mark_y = 20
# x_mark_size = 30
# close_area_size = 50
#
# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         height, width, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_frame)
#
#         x_pos = width - x_mark_x - x_mark_size
#         y_pos = x_mark_y
#
#         cv2.line(frame, (x_pos, y_pos), (x_pos + x_mark_size, y_pos + x_mark_size), (0, 0, 255), 3)
#         cv2.line(frame, (x_pos + x_mark_size, y_pos), (x_pos, y_pos + x_mark_size), (0, 0, 255), 3)
#
#         cv2.rectangle(frame,
#                       (x_pos - close_area_size // 2, y_pos - close_area_size // 2),
#                       (x_pos + x_mark_size + close_area_size // 2, y_pos + x_mark_size + close_area_size // 2),
#                       (0, 0, 255), 1)
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV uses BGR format, but MediaPipe requires RGB
#         result = hands.process(rgb_frame)
#
#         finger_in_close_area = False
#
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                 wrist = hand_landmarks.landmark[0]
#
#                 x_values = [lm.x for lm in hand_landmarks.landmark]
#                 y_values = [lm.y for lm in hand_landmarks.landmark]
#                 z_values = [lm.z for lm in hand_landmarks.landmark]
#
#                 min_x, max_x = min(x_values), max(x_values)
#                 min_y, max_y = min(y_values), max(y_values)
#
#                 box_width = max_x - min_x
#                 box_height = max_y - min_y
#                 max_abs_z = max(abs(z) for z in z_values)
#
#                 landmarks = []
#                 for lm in hand_landmarks.landmark:
#                     normalized_x = (lm.x - min_x) / box_width if box_width >0 else 0
#                     normalized_y = (lm.y - min_y) / box_height if box_height >0 else 0
#                     normalized_z = lm.z / max_abs_z if max_abs_z > 0 else 0
#                     landmarks.extend([normalized_x, normalized_y, normalized_z])
#
#                 # Print or use the normalized values
#                 # for i in range(0, len(normalized_values), 3):
#                 #     print(
#                 #         f"Normalized x: {normalized_values[i]:.4f}, y: {normalized_values[i + 1]:.4f}, z: {normalized_values[i + 2]:.4f}")
#
#
#                 # Index finger (landmark 8)
#
#                 index_finger_tip = hand_landmarks.landmark[8]
#                 index_finger_tip_position = (int(index_finger_tip.x * width),
#                                              int(index_finger_tip.y * height)) # converts the normalized coordinates into pixel coords
#
#                 cv2.circle(frame, index_finger_tip_position, 10, (0, 255, 0), -1)
#
#                 if (x_pos - close_area_size // 2 <= index_finger_tip_position[
#                     0] <= x_pos + x_mark_size + close_area_size // 2) and \
#                         (y_pos - close_area_size // 2 <= index_finger_tip_position[
#                             1] <= y_pos + x_mark_size + close_area_size // 2):
#                     finger_in_close_area = True
#
#                     cv2.line(frame, (x_pos, y_pos), (x_pos + x_mark_size, y_pos + x_mark_size), (255, 0, 0), 4)
#                     cv2.line(frame, (x_pos + x_mark_size, y_pos), (x_pos, y_pos + x_mark_size), (255, 0, 0), 4)
#
#                 key = cv2.waitKey(1) & 0xFF
#                 if (key in range(97, 123)) or (key in range(48, 58)):  # Letters (97-122) and Numbers (48-57)
#                     label = chr(key)
#                     csv_writer.writerow([label] + landmarks)
#                     print(f"Saved: {label}")
#
#         cv2.imshow("Hand Landmarks", frame)
#
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or finger_in_close_area:
#             print("Window closed by", "keyboard" if not finger_in_close_area else "finger gesture")
#             break
#
# cap.release()
# cv2.destroyAllWindows()
# csv_file.close()

import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

base_dir = "asl_data"
os.makedirs(base_dir, exist_ok=True)  # Ensure base directory exists

x_mark_x = 20
x_mark_y = 20
x_mark_size = 30
close_area_size = 50

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        x_pos = width - x_mark_x - x_mark_size
        y_pos = x_mark_y

        cv2.line(frame, (x_pos, y_pos), (x_pos + x_mark_size, y_pos + x_mark_size), (0, 0, 255), 3)
        cv2.line(frame, (x_pos + x_mark_size, y_pos), (x_pos, y_pos + x_mark_size), (0, 0, 255), 3)

        finger_in_close_area = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_values = [lm.x for lm in hand_landmarks.landmark]
                y_values = [lm.y for lm in hand_landmarks.landmark]
                z_values = [lm.z for lm in hand_landmarks.landmark]

                min_x, max_x = min(x_values), max(x_values)
                min_y, max_y = min(y_values), max(y_values)
                max_abs_z = max(abs(z) for z in z_values)

                # ✅ Corrected Syntax Here
                box_width = max_x - min_x
                box_height = max_y - min_y

                landmarks = []
                for lm in hand_landmarks.landmark:
                    normalized_x = (lm.x - min_x) / box_width if box_width > 0 else 0
                    normalized_y = (lm.y - min_y) / box_height if box_height > 0 else 0
                    normalized_z = lm.z / max_abs_z if max_abs_z > 0 else 0
                    landmarks.extend([normalized_x, normalized_y, normalized_z])

                index_finger_tip = hand_landmarks.landmark[8]
                index_finger_tip_position = (int(index_finger_tip.x * width),
                                             int(index_finger_tip.y * height))

                cv2.circle(frame, index_finger_tip_position, 10, (0, 255, 0), -1)

                if (x_pos - close_area_size // 2 <= index_finger_tip_position[0] <= x_pos + x_mark_size + close_area_size // 2) and \
                   (y_pos - close_area_size // 2 <= index_finger_tip_position[1] <= y_pos + x_mark_size + close_area_size // 2):
                    finger_in_close_area = True
                    cv2.line(frame, (x_pos, y_pos), (x_pos + x_mark_size, y_pos + x_mark_size), (255, 0, 0), 4)
                    cv2.line(frame, (x_pos + x_mark_size, y_pos), (x_pos, y_pos + x_mark_size), (255, 0, 0), 4)

                key = cv2.waitKey(1) & 0xFF
                if (key in range(97, 123)) or (key in range(48, 58)):  # Letters and numbers
                    label = chr(key)
                    folder_path = os.path.join(base_dir, label)
                    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

                    file_path = os.path.join(folder_path, f"{label}.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        if not file_exists:
                            csv_writer.writerow(['letter'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']])
                        csv_writer.writerow([label] + landmarks)

                    print(f"Saved: {label}")

        cv2.imshow("Hand Landmarks", frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or finger_in_close_area:
            print("Window closed by", "keyboard" if not finger_in_close_area else "finger gesture")
            break

cap.release()
cv2.destroyAllWindows()
