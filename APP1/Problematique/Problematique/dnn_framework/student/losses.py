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
        # N est le nombre de lot
        # C est le nombre de classe (10 dans le cas de la prob)

        N, C = x.shape

        # softmax forward
        x2, D = softmax(x)
        
        # CE forward
        real_target = np.eye(C)[target]
        L = -np.sum(real_target*np.log(x2))/N
        
        # CE backward
        dCE_dx = -real_target/x2

        # Backwards of Loss with respect to input prediction
        dL_dy = np.zeros_like(x)
        for n in range(N):
            dL_dy[n, :] = np.dot(D[n, :], dCE_dx[n, :])/N
        return (L , dL_dy)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """

    N, C = x.shape
    y = np.zeros_like(x)
    dSM_dx = np.zeros((N, C, C))
    for n in range(N):
        # forward
        y[n] = np.exp(x[n]) / np.sum(np.exp(x[n]))
        
        # D for backward
        D1 = y[n]*(1-y[n]) * np.eye(C)
        D2 = -np.outer(y[n], y[n])
        D2[np.diag_indices_from(D2)] = np.diag(D1)
        dSM_dx[n] = D2

    return (y, dSM_dx)


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
        dL = 2*(x - target)/x.size

        # dL fonctionne pas

        return (L, dL)
