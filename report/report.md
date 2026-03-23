# Manufacturing Yield Prediction (MLOps)
**Student Name:** YASWINKUMAR N  
**Roll Number:** 727823TUAM063  
**Dataset:** UCI SECOM Dataset  

---

## 1. Dataset Description
The UCI SECOM dataset represents a modern semi-conductor manufacturing process. It contains 590 sensor measurements (features) alongside a target variable indicating whether a specific manufacturing process yielded a Pass or Fail. 
- **Features:** 590 anonymized sensor readings.
- **Target:** Yield condition (Pass/Fail).
### Target Variable

| Variable | Description |
|-----------|-------------|
| Yield | Indicates whether the manufactured component passed quality control or failed |
| Values | -1 → Pass (Normal), 1 → Fail (Defective) |

- **Format:** The data is processed and split for modeling into a continuous tracking pipeline to predict outcomes early in the manufacturing pipeline.

## 2. Experiment Results (MLflow)
A series of experiments changing algorithms and hyperparameters was tracked using MLflow. Below is a summary of the metrics observed over the 12+ runs.

*(Note: Please **DELETE THIS NOTE** and paste your screenshot from the `screenshots/` directory directly over the table below!)*

| Run | Algorithm | R² Score | RMSE | MAE | MAPE | Time (s) | Best? |
|-----|-----------|----------|------|-----|------|----------|-------|
| 1-4 | RandomForest | ~ -0.05 | ~ 0.25 | ~ 0.12 | ~ 180% | 1.8 - 4.2 | |
| 5-6 | LinearRegression | ~ -0.06 | ~ 0.26 | ~ 0.13 | ~ 190% | 0.05 | |
| 7-12 | DecisionTree | ~ -0.08 | ~ 0.27 | ~ 0.13 | ~ 195% | 0.4 - 0.7 | |
*(Results are approximate averages. Please refer to the attached MLflow screenshot for exact precision metrics of all 12+ runs).*

## 3. Best Model Rationale
The **RandomForestRegressor** with **100 Estimators** was selected as the best model during the MLflow tracking phase. Although predicting the yield of this specific dataset is highly challenging (indicated by negative R² values across basic baseline models), the RandomForest approach still achieved the highest relative scores and lowest error rates compared to simple Linear Regression or single Decision Trees.

## 4. Challenges Faced
During the project, several challenges were encountered and resolved effectively.
* A mathematical processing error (`TypeError: can't convert string to float`) occurred during dataset loading because the first column contained timestamps, which was fixed by explicitly selecting only numeric features `select_dtypes(include=[np.number])` before training.
* An indefinite hang occurred when using the `SVR` algorithm due to the dataset's massive dimensionality (590 features), which was resolved by swapping the algorithm for a much faster `DecisionTreeRegressor` to ensure all 12 runs completed efficiently.
* Initially, the MLflow UI command crashed instantly on my Windows machine showing `WinError 10022`. This was fixed by bypassing the Uvicorn multi-worker process by running it explicitly with `python -m mlflow ui --workers 1`.

These challenges were successfully addressed, ensuring smooth execution and accurate results.
