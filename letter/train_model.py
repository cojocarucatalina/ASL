import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('asl_data.csv')

X = df.drop(columns=['letter'])
y = df['letter']

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for letter in df['letter'].unique():
    letter_data = df[df['letter'] == letter]
    X_letter = letter_data.drop(columns=['letter'])
    y_letter = letter_data['letter']

    if len(X_letter) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_letter, y_letter, test_size=0.2, random_state=42)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    else:
        X_train_list.append(X_letter)
        y_train_list.append(y_letter)

X_train_full = pd.concat(X_train_list, axis=0)
X_test_full = pd.concat(X_test_list, axis=0)
y_train_full = pd.concat(y_train_list, axis=0)
y_test_full = pd.concat(y_test_list, axis=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_full, y_train_full)

joblib.dump(clf, 'model.pkl')

print("Model trained and saved as 'model.pkl'")
