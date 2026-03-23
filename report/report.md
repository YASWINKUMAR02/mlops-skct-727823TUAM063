# Manufacturing Yield Prediction (MLOps)
**Student Name:** YASWINKUMAR N  
**Roll Number:** 727823TUAM063  
**Dataset:** UCI SECOM Dataset  

---

## 1. Dataset Description
The UCI SECOM dataset represents a modern semi-conductor manufacturing process. It contains 590 sensor measurements (features) alongside a target variable indicating whether a specific manufacturing process yielded a Pass or Fail. 
- **Features:** 590 anonymized sensor readings.
- **Target:** Yield condition (Pass/Fail).
- **Format:** The data is processed and split for modeling into a continuous tracking pipeline to predict outcomes early in the manufacturing pipeline.

## 2. Experiment Results (MLflow)
A series of experiments changing algorithms and hyperparameters was tracked using MLflow. Below is a summary table of the metrics observed over the 12+ runs.

*(Note: Replace this table with your actual output or just embed the screenshot from the `screenshots/` directory)*

| Run | Algorithm | Hyperparameters | MAE | RMSE | R² | MAPE | Time (s) | Best? |
|-----|-----------|-----------------|-----|------|----|------|----------|-------|
| 1   | RandomForest | n_estimators=50, max_depth=None | ... | ... | ... | ... | ... | |
| 2   | RandomForest | n_estimators=100, max_depth=None | ... | ... | ... | ... | ... | |
| 3   | RandomForest | n_estimators=50, max_depth=10 | ... | ... | ... | ... | ... | |
| 4   | RandomForest | n_estimators=100, max_depth=10 | ... | ... | ... | ... | ... | ⭐ |
| 5   | LinearReg | fit_intercept=True | ... | ... | ... | ... | ... | |
| 6   | LinearReg | fit_intercept=False | ... | ... | ... | ... | ... | |
| 7   | SVR | C=0.1, kernel='rbf' | ... | ... | ... | ... | ... | |
| 8   | SVR | C=1.0, kernel='rbf' | ... | ... | ... | ... | ... | |
| 9   | SVR | C=0.1, kernel='linear' | ... | ... | ... | ... | ... | |
| 10  | SVR | C=1.0, kernel='linear' | ... | ... | ... | ... | ... | |
| 11  | ... | ... | ... | ... | ... | ... | ... | |
| 12  | ... | ... | ... | ... | ... | ... | ... | |

## 3. Best Model Rationale
The **[Insert Algorithm Name]** with parameters **[Insert Params]** was selected as the best model because it achieved the highest explanatory power for the variance in the target variable (highest R² score) while maintaining a minimal Mean Absolute Error (MAE).

## 4. Encountered Error during Pipeline Setup
*(Replace with an actual error you faced. Below is an example.)*

During the setup of the local MLflow server, I initially encountered an `Exception: Experiment '...' already exists` error when the script tried to recreate my named experiment on subsequent runs. The traceback was:
```python
mlflow.exceptions.MlflowException: Experiment 'SKCT_727823TUAM063_UCI_SECOM_Dataset' already exists.
```
I resolved this issue by wrapping the `create_experiment` call inside a `try...except Exception:` block so that it safely passes if the experiment ID has already been registered in the `mlruns` directory.
