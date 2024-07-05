import logging
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
import pandas as pd
from src.data_cleaning import DataPreProcessingStrategy, DataCleaning, DataDivideStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        # 1-step
        preprocessingStrategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, preprocessingStrategy)
        process_data = data_cleaning.handle_data()
        logging.info(process_data.describe())

        # 2-step
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.info(e)
        raise e

