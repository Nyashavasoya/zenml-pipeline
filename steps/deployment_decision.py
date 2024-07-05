from zenml import step
from zenml.steps import BaseParameters, Output

class DeploymentTrigger(BaseParameters):
    min_accuracy: float = 0.92

@step
def deployment_trigger(accuracy: float, config: DeploymentTrigger):
    return accuracy > config.min_accuracy
