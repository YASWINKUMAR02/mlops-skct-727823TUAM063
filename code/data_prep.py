# YASWINKUMAR N - Roll No: 727823TUAM063
import os
import pandas as pd
from datetime import datetime

print(f"Roll Number: 727823TUAM063 - Timestamp: {datetime.now().isoformat()}")

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "secom.csv")  # adjust as needed
OUTPUT_TRAIN = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
OUTPUT_TEST = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")

# Load raw dataset
df = pd.read_csv(DATA_PATH)
# Simple train-test split (80/20)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
# Save splits
train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)
print("Data split saved: train.csv and test.csv")
