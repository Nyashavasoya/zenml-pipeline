import logging

from zenml import step
import pandas as pd
import numpy as np
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"]
]:
    try:
        y_pred = model.predict(X_test)
        mse_class = MSE()
        y_test = y_test.to_numpy()
        mse = mse_class.calculate_score(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric(key="mse", value=mse)

        r2_class = R2()
        r2_score = r2_class.calculate_score(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric("r2_score", r2_score)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric("rmse", rmse)

        return r2_score, rmse
    except Exception as e:
        logging.info(e)
        raise e



