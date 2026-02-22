from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
import mlflow.sklearn
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def create_experiment(experiment_name):
    """Sets the active experiment. Creates it if it doesn't exist."""
    mlflow.set_experiment(experiment_name)

def start_run(run_name=None):
    """Starts a new MLflow run and returns the run object."""
    return mlflow.start_run(run_name=run_name)

def log_dataset(input, output, name, context):
    """Logs basic metadata about the landmarks dataset."""
    dataset = input.copy()
    dataset['target'] = output.values
    dataset = mlflow.data.from_pandas(dataset, targets='target', name=name) 
    mlflow.log_input(dataset, context=context)

def log_parameters(params_dict):
    """Logs a dictionary of hyperparameters or preprocessing settings."""
    mlflow.log_params(params_dict)

def log_metrics(metrics_dict):
    """Logs evaluation metrics (Accuracy, Precision, Recall, F1)."""
    mlflow.log_metrics(metrics_dict)

def log_artifact(local_path, artifact_path=None):
    """Logs a local file or directory as an artifact in the current MLflow run."""
    mlflow.log_artifact(local_path, artifact_path)

def get_model_signature(X_train, model):
    """
    Creates a model signature which defines the schema of 
    the model's inputs and outputs.
    """
    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)
    return signature

def log_model(model, artifact_path, signature):
    """
    Logs the model to the current MLflow run with its signature.
    Does NOT register it yet.
    """
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature
    )

def register_model(run_id, artifact_path, model_name):
    """
    Registers a specific model from a completed run into the 
    MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version = mlflow.register_model(model_uri, model_name)
    return version

def set_model_description(model_name, description):
    """Adds a detailed description to the registered model."""
    client = MlflowClient()
    client.update_registered_model(
        name=model_name,
        description=description
    )

def set_model_alias(model_name, alias, version):
    """Assigns an alias (e.g., 'Champion') to a specific model version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)

def set_model_version_details(model_name, version, description):
    """Adds a description specifically to a specific version of the model."""
    client = MlflowClient()
    client.update_model_version(
        name=model_name,
        version=version,
        description=description
    )

def evaluate_model(model, model_type, X_test, y_test):
    """Evaluates the trained model and returns a dictionary of metrics."""

    # Get the predictions
    y_pred = model.predict(X_test)

    # Create a DataFrame for the evaluation
    eval_data = pd.DataFrame(y_pred, columns=['predictions'])
    eval_data['targets'] = y_test.values

    result = mlflow.evaluate(data=eval_data, model_type=model_type, predictions='predictions', targets='targets')

    return result.metrics

def end_run():
    """Ends the current active run."""
    mlflow.end_run()