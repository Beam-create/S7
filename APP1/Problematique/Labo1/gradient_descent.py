import numpy as np
import matplotlib.pyplot as plt

"""
We look to minimize the equation : Loss = ||BA - I||2_2
The notation above indicates that all the elements of the matrix BA are squared and summed.
i.e. ||X||2_2 = sum(Xij^2)

Here the gradient of the loss function is given by : dL/dB = 2(BA - I)A^T
We are looking to optimize the B matrix using an iterative method, defined by the following equation :
B(k) = B(k-1) - alpha * dL/dB, where alpha is the learning rate.
"""

# Matrix inversion
A = np.array([[3, 4, 1], 
              [5, 2, 3],
              [6, 2, 2]])

def matrix_inversion(A, alpha, iterations):
   for i in range(iterations):
      B = np.random.rand(3, 3)
      loss = []
      for j in range(iterations):
         loss.append(np.linalg.norm(B.dot(A) - np.identity(3), ord=2))
         B = B - alpha * 2 * (B.dot(A) - np.identity(3)).dot(A.T)
      plt.plot(loss, label='B{}'.format(i))