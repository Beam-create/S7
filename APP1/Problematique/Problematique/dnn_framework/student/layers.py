import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.I = input_count
        self.J = output_count

        # We want reproducible results
        manual_seed = 3224
        np.random.seed(manual_seed)
        self.W = np.random.randn(self.J, self.I) * (2/(self.I + self.J))
        self.b = np.random.randn(1, self.J) * (2/self.J)
        self.params = {"w" : self.W, "b" : self.b}
        

    def get_parameters(self):
        return self.params

    def get_buffers(self):
        return {}

    def forward(self, x):
        Y = (self.W @ x.T).T + self.b
        return (Y, x)

    def backward(self, output_grad, cache):
        dX = (output_grad @ self.W)
        dW = (output_grad.T@cache)
        db =  np.sum(output_grad, 0)

        grad = {"w" : dW , "b" : db}

        return (dX, grad)


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.gamma = np.ones((input_count))
        self.beta = np.zeros((input_count))
        self.alpha = alpha
        self.eps = 1e-10

        self.global_mean = np.empty((input_count))
        self.global_variance = np.empty((input_count))

        self.params = {'gamma' : self.gamma, 'beta' : self.beta}
        self.buffer = {'global_mean' : self.global_mean, 
                       'global_variance' : self.global_variance}

    def get_parameters(self):
        return self.params

    def get_buffers(self):
        return self.buffer

    def forward(self, x):

        if self._is_training :
            return self._forward_training(x)
        
        else :
            return self._forward_evaluation(x)
        
    def _forward_training(self, x):
        
        mean_B = np.mean(x, 0)
        var_B = np.var(x, 0)

        self.global_mean = (1-self.alpha)*self.global_mean + self.alpha*mean_B
        self.global_variance = (1-self.alpha)*self.global_variance + self.alpha*var_B
        
        x_est = (x - mean_B)/np.sqrt(var_B + self.eps)

        y = (self.gamma*x_est + self.beta)
        return (y, (x, x_est, mean_B, var_B))

    def _forward_evaluation(self, x):

        avg = self.global_mean
        var = self.global_variance

        x_est = (x - avg)/np.sqrt(var  + self.eps)

        y = (self.gamma*x_est + self.beta)
        return (y, (x, x_est, avg, var))

    def backward(self, output_grad, cache):
        # Extraire la cache
        x, x_est, mean, var = cache
        N, M = x.shape

        # Calcul des grad
        dx_est = output_grad * self.gamma
        dvar = np.sum(dx_est*(x-mean)*(-1/2)*(var+self.eps)**(-3/2), 0)
        dmean = -np.sum(dx_est, 0)/np.sqrt(var+self.eps)
        dx = dx_est/np.sqrt(var+self.eps) + (2/N) * dvar * (x-mean) + (1/N)*dmean
        dgamma = np.sum(output_grad*x_est, 0)
        dbeta = np.sum(output_grad, 0)

        grads = {'gamma' : dgamma, 'beta' : dbeta}

        return (dx, grads)

class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        params = {}
        return params

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1/( 1 + np.exp(-x))
        return (y, y)

    def backward(self, output_grad, cache):
        return ((1-cache)*cache * output_grad, {})


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        params = {}
        return params

    def get_buffers(self):
        return {}

    def forward(self, x):
        return (np.maximum(0,x), x)

    def backward(self, output_grad, cache):
        return ((cache>0)*output_grad, {})
