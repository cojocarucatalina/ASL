import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

DATA_SAVE_PATH = "landmarks_data.pkl"
MODEL_SAVE_PATH = "asl_model_random_forest.pkl"

def train_random_forest():
    """Load extracted features, train Random Forest model, and save it."""
    # Load pre-extracted data
    with open(DATA_SAVE_PATH, 'rb') as f:
        X, y = pickle.load(f)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_random_forest()
