import numpy as np

import sys
sys.path.append('C:\\Users\\Mathieu\\Documents\\UNIVERSITÉ\\COURS\\S7\\github\\S7\\APP1\\Problematique\\Problematique')
sys.path.append('C:\\Users\\fulld\\Documents\\UNIVERSITÉ\\COURS\\S7\\github\\S7\\APP1\\Problematique\\Problematique')


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
        dx2 = -real_target/x2

        # softmax backward
        dx = np.zeros_like(x)
        for n in range(N):
            dx[n, :] = np.dot(D[n, :], dx2[n, :])/N

        return (L , dx)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """

    N, C = x.shape
    y = np.zeros_like(x)
    D = np.zeros((N, C, C))
    for n in range(N):
        # forward
        y[n] = np.exp(x[n]) / np.sum(np.exp(x[n]))
        
        # D for backward
        D1 = y[n]*(1-y[n]) * np.eye(C)
        D2 = -np.outer(y[n], y[n])
        D2[np.diag_indices_from(D2)] = np.diag(D1)
        D[n] = D2

    return (y, D)


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
