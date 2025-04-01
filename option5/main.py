import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('letter_recognition_model.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    img = cv2.resize(frame, (64, 64))  # Resize to the same size as training
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the letter
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Map predicted_class back to the letter (0 -> A, 1 -> B, ..., 25 -> Z)
    letter = chr(predicted_class + ord('A'))

    # Display the letter on the image
    cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Live Letter Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

