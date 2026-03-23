import mlflow
import os

def init_mlflow(experiment_name: str, tracking_uri: str = "file://./mlruns"):
    """Initialize MLflow tracking server and set the experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI for the tracking server (default local file store).
    """
    # Set the tracking URI (local file store)
    mlflow.set_tracking_uri(tracking_uri)
    # Create the experiment if it does not exist
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        # Experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    # Set the active experiment
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized. Experiment '{experiment_name}' (ID: {experiment_id}) is active.")
    return experiment_id

if __name__ == "__main__":
    # Example usage – replace with actual values or import from another module
    EXPERIMENT_NAME = "SKCT_727823TUAM063_UCI_SECOM_Dataset"
    init_mlflow(EXPERIMENT_NAME)
