from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.params = parameters
        self.mu = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        return parameter - self.mu*parameter_grad
