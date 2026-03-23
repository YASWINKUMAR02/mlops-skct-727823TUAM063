# YASWINKUMAR N - Roll No: 727823TUAM063
import os
import pandas as pd
from datetime import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"Roll Number: 727823TUAM063 - Timestamp: {datetime.now().isoformat()}")

# Paths
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rf_model.pkl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation.txt")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# Load test data (last column is target)
df = pd.read_csv(TEST_PATH)
X_test = df.iloc[:, :-1]
y_true = df.iloc[:, -1]

# Load model
model = joblib.load(MODEL_PATH)

# Predict and compute metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

# Write results
with open(RESULTS_PATH, "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
print(f"Evaluation results saved to {RESULTS_PATH}")
