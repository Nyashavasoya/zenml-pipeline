from zenml.steps import BaseParameters

class ModelNameConfiguration(BaseParameters):
    model_name: str = 'linear_regression'
