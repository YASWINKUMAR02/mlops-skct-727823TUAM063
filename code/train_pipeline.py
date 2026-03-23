# YASWINKUMAR N - Roll No: 727823TUAM063
import os
import pandas as pd
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor

print(f"Roll Number: 727823TUAM063 - Timestamp: {datetime.now().isoformat()}")

# Paths
TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rf_model.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load training data (last column is target)
df = pd.read_csv(TRAIN_PATH)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train a simple model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model artifact
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
