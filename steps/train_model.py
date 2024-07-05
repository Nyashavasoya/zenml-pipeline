from zenml import step
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfiguration
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                config: ModelNameConfiguration) -> RegressorMixin:
    model = None
    if config.model_name == "linear_regression":
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.model_train(X_train=X_train, y_train=y_train)
        return trained_model
    else:
        raise ValueError("Error")
