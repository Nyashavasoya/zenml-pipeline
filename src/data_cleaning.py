import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessingStrategy(DataStrategy):

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_columns = df.select_dtypes(include='number')
            df = numeric_columns
            for col in df.columns:
                # Convert the column to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            logging.info(df.info())
            logging.info(df.describe())
            return df
        except Exception as e:
            logging.info(e)
            raise e


class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        try:
            logging.info(data.describe())
            logging.info(data.info())
            X = data.drop("review_score", axis=1)
            logging.info(X.describe())
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:

    def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
        self.data = df
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.info(e)
            raise e
