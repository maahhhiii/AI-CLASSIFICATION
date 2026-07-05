import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_data

# Load dataset
df = load_data()

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

# Save model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, MODEL_PATH)

print(f"Model saved successfully at:\n{MODEL_PATH}")