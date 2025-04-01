import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_random_forest(features_file='features_data.pkl', output_file='forest.pkl'):
    """Train a Random Forest classifier on the extracted features"""
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

    # Create and train the Random Forest
    print("Training Random Forest classifier...")
    forest = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    forest.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = forest.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Get feature importance
    feature_importances = forest.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    print("\nTop 10 most important features:")
    for i in range(min(10, len(feature_importances))):
        print(f"Feature {indices[i]}: {feature_importances[indices[i]]:.4f}")

    # Save the model and scaler
    with open(output_file, 'wb') as f:
        pickle.dump({'model': forest, 'scaler': scaler, 'classes': list(range(len(np.unique(y))))}, f)

    print(f"Random Forest model saved to {output_file}")

    return forest, scaler


if __name__ == "__main__":
    model, scaler = train_random_forest()
    print("Random Forest training complete")