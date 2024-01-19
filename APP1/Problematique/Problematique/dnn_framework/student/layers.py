import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.I = input_count
        self.J = output_count

        self.W = np.ones((self.J, self.I))
        self.b = np.ones((1,self.J))

        self.params = {"w" : self.W, "b" : self.b}

    def get_parameters(self):
        return self.params

    def get_buffers(self):
        raise NotImplementedError()

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
        self.gamma = np.ones((input_count))
        self.beta = np.zeros((input_count))
        self.alpha = alpha

        self.global_mean = 0
        self.global_variance = 0

        self.params = {'gamma' : self.gamma, 'beta' : self.beta}
        self.buffer = {'global_mean' : self.global_mean, 
                       'global_variance' : self.global_variance}

    def get_parameters(self):
        return self.params

    def get_buffers(self):
        return self.buffer

    def forward(self, x):
        avg =np.mean(x, 0)
        dev = np.std(x, 0)

        self.global_mean = (1-self.alpha)*self.global_mean + self.alpha*avg
        self.global_variance = (1-self.alpha)*self.global_variance + self.alpha*dev
        
        avg = self.global_mean
        dev = self.global_variance
        if dev.any == 0:
            dev = 10e-6
        x_est = (x - avg)/np.sqrt(dev**2)

        y = (self.gamma.T@x_est.T + self.beta).T


        return (y, x_est)

    def _forward_training(self, x):
        return self.forward(x)

    def _forward_evaluation(self, x):
        return self.forward(x)

    def backward(self, output_grad, cache):
        dx = output_grad@self.gamma
        dgamma = np.sum(output_grad@cache.T)
        dbeta = np.sum(output_grad)

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
        raise NotImplementedError()

    def forward(self, x):
        y = 1/( 1 + np.exp(-x))
        return (y, y)

    def backward(self, output_grad, cache):
        return ((1-cache)*cache * output_grad, None)


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        params = {}
        return params

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        return (np.maximum(0,x), x)

    def backward(self, output_grad, cache):
        return ((cache>0)*output_grad, cache)
