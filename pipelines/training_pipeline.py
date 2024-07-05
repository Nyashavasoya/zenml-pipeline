import logging

import pandas as pd
from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
import numpy as np
import pandas as pd


@pipeline
def training_pipeline(data_path: str) -> None:
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    logging.info(type(X_train))
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model=model, X_test=X_test, y_test=y_test)
