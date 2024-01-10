import numpy as np
import matplotlib.pyplot as plt

def computeLoss(y_est, y):
    # Compute the loss function
    L = np.sum( (y_est - y)**2 )

    # return the loss value
    return L

def computeYEstimate(a, xi):
    
    y_est = a@xi

    return y_est

def computexi(x : np.array, N : int):
    
    I = len(x)
    xi = np.zeros((N+1, I))

    for i in range(0, N+1):
        xi[i] = x**i

    return xi

def computeGradient(y_est, y, xi, N):
    # Compute the gradient matrix
    G = 2 * (y_est - y)@xi.T

    #G = 2* np.sum( (y_est - y)@xi.T )

    # return the gradient matrix
    return G

def optmizeByIterationNumber(x, y, N, iteration, step):
    a = np.zeros(N+1)
    xi = computexi(x, N)
    y_est = computeYEstimate(a, xi)
    L = computeLoss(y_est, y)
    Larray = []
    G = computeGradient(y_est, y, xi, N)
    
    anext = a
    for i in range (iteration):
        G = computeGradient(y_est, y, xi, N)
        anext = a - step*G
        a = anext
        y_est = computeYEstimate(a, xi)
        L = computeLoss(y_est, y)
        Larray.append(L)

    plt.plot(Larray)
    plt.show()

    return L, a



x = np.array([-0.95,-0.82,-0.62,-0.43,-0.17,-0.07,0.25,0.38,0.61,0.79,1.04])
y = np.array([0.02,0.03,-0.17,-0.12,-0.37,-0.25,-0.10,-0.14,0.53,0.71,1.53])


# Find vector by optimization
iteration = 1000
step = 0.001
N = 7
L, a = optmizeByIterationNumber(x, y, N, iteration, step)

print(L)
print(a)

# Extrapolate solution between [-1.25, 1.25]
x_ext = np.linspace(-1.25, 1.25, 100)
x_ext_i = computexi(x_ext, N)
y_ext = computeYEstimate(a, x_ext_i)

# Show initial data
plt.scatter(x, y)
plt.plot(x_ext, y_ext, color='red')
plt.show()
