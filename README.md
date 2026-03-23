# SKCT_727823TUAM063_UCI_SECOM_Dataset

**Student Name:** YASWINKUMAR N
**Roll Number:** 727823TUAM063
**Dataset:** UCI SECOM Dataset

## Step-by-Step Execution Guide

### 1. Prerequisites and Setup
1. Open your terminal in the `C:\727823TUAM063_MLOps` folder.
2. Ensure you have installed the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *(Requirements: mlflow, scikit-learn, pandas, numpy, matplotlib, jupyter, joblib)*

### 2. Prepare the Data
1. Create a `data` folder inside `C:\727823TUAM063_MLOps`.
2. Download the **UCI SECOM dataset** and save it as `secom.csv` in the `data/` folder.
   - *Note: Ensure the target column (Yield) is the last column in the CSV.*

### 3. Component A: MLflow Experiment Tracking
1. First, we need to make sure the training script is in the right place. Rename `run_experiments.py` to `training.py` and move it into the `code` folder.
2. Look inside `code/training.py` and update the `DATA_PATH` to point to `../data/secom.csv`.
3. Run the script:
   ```bash
   python code/training.py
   ```
   - This will load the data, save 3 EDA plots (in the `eda/` folder), and run 12+ ML models.
   - It will log the metrics (MAE, RMSE, R2, MAPE, training time, size) and tags to local MLflow.
4. Start the MLflow UI to view your results:
   ```bash
   mlflow ui
   ```
   - Open your browser to `http://localhost:5000`.
   - Take a screenshot of the 12+ runs for your `screenshots/` folder.

### 4. Component B: Azure ML Pipeline (Local Testing)
The pipeline is divided into 3 stages. Test them sequentially:
1. **Prepare Data:**
   ```bash
   python code/data_prep.py
   ```
   *(Creates `train.csv` and `test.csv` in the `data/` folder)*
2. **Train Model:**
   ```bash
   python code/train_pipeline.py
   ```
   *(Creates `rf_model.pkl` in the `model/` folder)*
3. **Evaluate Model:**
   ```bash
   python code/evaluate.py
   ```
   *(Creates `evaluation.txt` in the `results/` folder)*
4. **Submit to Azure** (Optional/If required by your setup):
   Use the CLI to submit your YAML:
   ```bash
   az ml job create -f code/pipeline_727823TUAM063.yml
   ```
   - Take a screenshot of the Azure Portal showing the pipeline run.

### 5. Component C: Final Submission Prep
1. **EDA Notebook:** Create `notebooks/eda.ipynb` and copy the basic data loading and plotting code into it. Run all cells fresh.
2. **Report:** Write your 2-page report (`report/report.pdf`) including your dataset details, MLflow results table, and a paragraph about an error you encountered.
3. **Git History:** 
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
   - Make sure to create at least 7 commits across 3 different days as per the requirement.
4. **ZIP the files:** Ensure your folder structure matches the assignment EXACTLY. Zip the files as `727823TUAM063_MLOps.zip`.
////