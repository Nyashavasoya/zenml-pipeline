import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class Model(ABC):

    @abstractmethod
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass


class LinearRegressionModel(Model):

    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        try:
            # logging.info(X_train.shape)
            logging.info(X_train.select_dtypes(include='number'))
            # logging.info(X_train.columns)
            reg = LinearRegression(**kwargs)
            reg.fit(X=X_train, y=y_train)
            return reg
        except Exception as e:
            logging.info(e)
            raise e
