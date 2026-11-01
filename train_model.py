
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv("predicta_dataset_v2.csv")

# 2. Select input features and target
X = data[["Vibration", "Temperature"]]
y = data["Health"]

# IMPORTANT: Do not use "Level" as an input.
# It may directly encode the target label.

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
labels = model.classes_
print("Class order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))

# 6. Save the trained model
joblib.dump(model, "motor_model.joblib")

print("\nModel saved as motor_model.joblib")
