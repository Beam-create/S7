import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        c = x
        a = np.log(c)
        b = target*a.T
        d = -np.sum(b)
        e = -np.mean(b)
        d2 = np.sum(-b)
        e2 = np.mean(-b)
        L = -np.sum(target*np.log(x.T))
        dL = -target/x.T

        # L et dL fonctionne pas

        return (L , dL)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    return np.exp(x) / np.sum(np.exp(x))


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        L = np.mean((x - target)**2)
        dL = 2*(x - target)

        # dL fonctionne pas

        return (L, dL)
