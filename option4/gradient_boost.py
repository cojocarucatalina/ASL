import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_gradient_boosting(features_file='features_data.pkl', output_file='gradient_boost.pkl'):
    """Train a Gradient Boosting classifier on the extracted features"""
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

    # Create and train the Gradient Boosting model
    print("Training Gradient Boosting classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        max_features=None,
        random_state=42,
        verbose=1
    )

    gb.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = gb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Get feature importance
    feature_importances = gb.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    print("\nTop 10 most important features:")
    for i in range(min(10, len(feature_importances))):
        print(f"Feature {indices[i]}: {feature_importances[indices[i]]:.4f}")

    # Save the model and scaler
    with open(output_file, 'wb') as f:
        pickle.dump({'model': gb, 'scaler': scaler, 'classes': list(range(len(np.unique(y))))}, f)

    print(f"Gradient Boosting model saved to {output_file}")

    return gb, scaler


if __name__ == "__main__":
    model, scaler = train_gradient_boosting()
    print("Gradient Boosting training complete")