from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client
from pipelines.training_pipeline import training_pipeline


if __name__ == '__main__':
    print(get_tracking_uri())
    training_pipeline(data_path="D:\\MLOps\\customer-satisfaction\\data\\olist_customers_dataset.csv")
