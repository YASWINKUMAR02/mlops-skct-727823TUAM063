# YASWINKUMAR N - Roll No: 727823TUAM063
import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

print(f"Roll Number: 727823TUAM063 - Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")

# ------------------------------------------------------------
# Configuration (replace paths as needed)
# ------------------------------------------------------------
ROLL_NUMBER = "727823TUAM063"
STUDENT_NAME = "YASWINKUMAR N"
DATASET_NAME = "UCI_SECOM_Dataset"
EXPERIMENT_NAME = f"SKCT_{ROLL_NUMBER}_{DATASET_NAME}"

# Adjusted paths since this is running from from 'code/' directory
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "secom.csv")
EDA_DIR = os.path.join(os.path.dirname(__file__), "..", "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def init_mlflow():
    # Set tracking URI to sqlite DB in root directory
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db')).replace('\\', '/')
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    try:
        mlflow.create_experiment(EXPERIMENT_NAME)
    except Exception:
        pass
    mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(path):
    # Depending on SECOM format, maybe space-separated. Assuming CSV here.
    df = pd.read_csv(path)
    # Just in case there are NaNs, fill with 0 for this basic run
    df = df.fillna(0)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number])
    y = df.iloc[:, -1]
    # If y is categorical (Pass/Fail) but we use Regressor, convert to numeric
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def perform_eda(X, y, out_dir):
    plt.figure()
    y.hist(bins=30)
    plt.title("Target Distribution")
    plt.savefig(os.path.join(out_dir, "target_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    # Sample correlation for speed
    corr = X.iloc[:, :20].corr().abs()
    plt.imshow(corr, cmap="viridis", aspect="auto")
    plt.title("Feature Correlation Heatmap (sample)")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"))
    plt.close()
    
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, alpha=0.5)
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.title("Feature 1 vs Target")
    plt.savefig(os.path.join(out_dir, "feature1_target.png"))
    plt.close()

def get_models_and_params():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
    }
    param_grid = {
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
        },
        "LinearRegression": {
            "fit_intercept": [True, False],
        },
        "DecisionTree": {
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
    }
    return models, param_grid

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero for MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def model_size_mb(model):
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell() / (1024 * 1024)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}.")
        print("Please place the SECOM dataset there as secom.csv")
        return

    init_mlflow()
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    
    print("Performing EDA...")
    perform_eda(X_train, y_train, EDA_DIR)

    models, param_grid = get_models_and_params()
    runs = []
    best_r2 = -np.inf
    best_run_id = None
    best_model = None

    print("Starting MLflow experiments...")
    for model_name, ModelClass in models.items():
        grid = list(ParameterGrid(param_grid[model_name]))
        for params in grid:
            with mlflow.start_run(run_name=f"{model_name}_{len(runs)+1}") as run:
                mlflow.set_tag("student_name", STUDENT_NAME)
                mlflow.set_tag("roll_number", ROLL_NUMBER)
                mlflow.set_tag("dataset", DATASET_NAME)
                mlflow.log_params(params)
                
                seed = np.random.randint(0, 1_000_000)
                np.random.seed(seed)
                mlflow.log_param("random_seed", seed)
                
                start = time.time()
                model = ModelClass(**params)
                model.fit(X_train, y_train)
                training_time = time.time() - start
                
                preds = model.predict(X_test)
                metrics = compute_metrics(y_test, preds)
                
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time_seconds", training_time)
                size_mb = model_size_mb(model)
                mlflow.log_metric("model_size_mb", size_mb)
                
                mlflow.sklearn.log_model(model, "model")
                if metrics["R2"] > best_r2:
                    best_r2 = metrics["R2"]
                    best_run_id = run.info.run_id
                    best_model = model
                runs.append({"run_id": run.info.run_id, "model": model_name, "params": params, "metrics": metrics})
                print(f"Run {len(runs)} completed: {model_name} (R2: {metrics['R2']:.4f})")

    if best_model is not None:
        best_path = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pkl")
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        joblib.dump(best_model, best_path)
        mlflow.log_artifact(best_path, "best_model")
        print(f"\nBest model (R2={best_r2:.4f}) saved to {best_path} (run ID: {best_run_id})")

if __name__ == "__main__":
    main()
