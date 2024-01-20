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
        # Utiliser le softmax
        # Au pire aller voir pytorch
        x2, D = softmax(x)
        real_target = np.eye(x.shape[1])[target]
        L = -np.sum(real_target*np.log(x2))/x.shape[0]
        
        dy = (-real_target/x)/x.shape[0]
        dx = np.mean(D@dy.T, 0).T
        # L et dL fonctionne pas

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
        y[n] = np.exp(x[n]) / np.sum(np.exp(x[n]))
        D1 = y[n]*(1-y[n]) * np.eye(C)
        D2 = -y[n]*y[n].T * np.ones((C,C))
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
