import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

DATA_SAVE_PATH = "landmarks_data.pkl"
MODEL_SAVE_PATH = "asl_model_neural_network.h5"

def train_neural_network():
    """Load extracted features, train Neural Network, and save it."""
    # Load pre-extracted data
    with open(DATA_SAVE_PATH, 'rb') as f:
        X, y = pickle.load(f)

    # One-hot encode the labels (A-Z -> 0-25)
    y = [ord(label) - ord('A') for label in y]  # Convert labels (A-Z) to integers (0-25)
    y = to_categorical(y, num_classes=26)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = Sequential([
        Flatten(input_shape=(X_train.shape[1],)),  # Flatten the input layer
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')  # 26 output classes (A-Z)
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy[1] * 100:.2f}%")

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_neural_network()
