import numpy as np
import matplotlib.pyplot as plt

def computeLoss(A, B, I):
    # Compute the loss function
    L = 0
    L1 = B*A-I
    for i in range(len(L1[0])):
        for j in range(len(L1[1])):
            L += L1[i, j]*L1[i,j]

    # return the loss value
    return L

def computeGradient(A, B, I):
    # Compute the gradient matrix
    G = 2*(B*A-I)*np.transpose(A)

    # return the gradient matrix
    return G

def optmizeByIterationNumber( step, iteration, A, B, I):
    
    L = computeLoss(A, B, I)
    Larray = []
    for i in range(iteration):
        G = computeGradient(A, B, I)
        Bnext = B - step*G
        B = Bnext
        L = computeLoss(A, B, I)
        Larray.append(L)

    plt.figure(figsize=(8, 6))
    plt.plot(Larray)
    plt.title('Graph of Loss value over the numer of iteration')
    plt.xlabel('Index')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.show()
        
    return L, B

def optmizeByErrorLimit( step, limit, A, B, I):
    
    L = computeLoss(A, B, I)
    Larray = []
    while L > limit:
        G = computeGradient(A, B, I)
        Bnext = B - step*G
        B = Bnext
        L = computeLoss(A, B, I)
        Larray.append(L)

    plt.figure(figsize=(8, 6))
    plt.plot(Larray)
    plt.title('Graph of Loss value over the number of iteration')
    plt.xlabel('Index')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.show()
        
    return L, B

def doTheLab(A, step, iteration, limit):
    B = np.ones(np.shape(A))

    I = np.identity(np.shape(A)[0])

    L1, B1 = optmizeByIterationNumber(step, iteration, A, B, I)

    print("Computed identity : ", B1*A)

A = np.array([[3,4,1],
              [5,2,3],
              [6,2,2]])

A2 = np.array([[3,4,1,2,1,5],
              [5,2,3,2,2,1],
              [6,2,2,6,4,5],
              [1,2,1,3,1,2],
              [1,5,2,3,3,3],
              [1,2,2,4,2,1]])

A3 = np.array([[2,1,1,2],
              [1,2,3,2],
              [2,1,1,2],
              [3,1,4,1]])

errorLim = 1e-12
step = 0.01
iteration = 1000

print("Starting process")
doTheLab(A3, step, iteration, errorLim)









