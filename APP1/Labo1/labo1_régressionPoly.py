import numpy as np
import matplotlib.pyplot as plt

def computeLoss(y_est, y):
    # Compute the loss function
    L = 0
    
    for i in range(len(y_est)):
        L += (y_est[i]-y[i])**2

    # return the loss value
    return L

def computeYEstimate(a, xi):
    xT = np.transpose(xi)

    y_est = a*xT

    return y_est

def computexi(x, N):
    xi = np.zeros(N+1)
    for i in range (N):
        xi[i] = x**i

    return xi

def computeGradient(y_est, y, xi):
    # Compute the gradient matrix
    for i in range(len(x)):
        G += (y_est[i]-y[i])*xi[i]

    G = 2*G

    # return the gradient matrix
    return G

N = 1
x = np.array([-0.95,-0.82,-0.62,-0.43,-0.17,-0.07,0.25,0.38,0.61,0.79,1.04])
y = np.array([0.02,0.03,-0.17,-0.37,-0.25,-0.10,-0.14,0.53,0.71,1.53])

a = np.zeros(N+1)
xi = computexi(x, N)
y_est = computeYEstimate(a, xi)
L = computeLoss(y_est, y)
while L > 0.1:
    pass