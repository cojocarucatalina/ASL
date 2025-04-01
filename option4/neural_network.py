import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_neural_network(features_file='features_data.pkl', output_file='nn.pkl'):
    """Train a neural network classifier on the extracted features"""
    # Load features and labels
    with open(features_file, 'rb') as f:
        data = pickle.load(f)

    X = data['features']
    y = data['labels']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine input dimensions and output classes
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))

    print(f"Training neural network...")
    print(f"Input dimensions: {input_dim}")
    print(f"Number of classes: {num_classes}")

    # Create and train the neural network
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # You can adjust the architecture
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=300,
        early_stopping=True,
        verbose=True,
        random_state=42
    )

    nn.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = nn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    with open(output_file, 'wb') as f:
        pickle.dump({'model': nn, 'scaler': scaler, 'classes': list(range(num_classes))}, f)

    print(f"Neural network model saved to {output_file}")

    return nn, scaler


if __name__ == "__main__":
    model, scaler = train_neural_network()
    print("Neural network training complete")