import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
class Evaluation:
    @abstractmethod
    def calculate_score(self):
        pass

class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            return mse
        except Exception as e:
            logging.info(e)
            raise e


class R2(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            return r2
        except Exception as e:
            logging.info(e)
            raise e

class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
            return rmse
        except Exception as e:
            logging.info(e)
            raise e
