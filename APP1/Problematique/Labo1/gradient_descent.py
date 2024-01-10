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

def matrix_inversion(A, alpha, iterations):
   seed_value = 42
   np.random.seed(seed_value)
   loss=[]
   B = np.random.rand(3, 3)
   for i in range(iterations):
      loss.append(np.linalg.norm(B.dot(A) - np.identity(3), ord=2))
      B = B - alpha * 2 * (B.dot(A) - np.identity(3)).dot(A.T)
      plt.plot(loss)
   print(B)
   print(np.invert(A))
   return B
   
   
if __name__ == "__main__":
   # Q1
   A = np.array([[3, 4, 1], 
              [5, 2, 3],
              [6, 2, 2]])
   
   B1 = matrix_inversion(A, 0.005, 1000)
   B2 = matrix_inversion(A, 0.001, 1000)
   B3 = matrix_inversion(A, 0.01, 1000)
   
   # Get the distance between the matrices for indices
   # Create 3 dimension array, (3, 3, 3). first dim is the matrix, second and third are the indices
   b_12 = np.zeros((3, 3))
   for j in range(3):
      for k in range(3):
         b_12[j][k] = abs(B1[j][k] - B2[j][k])
   b_23 = np.zeros((3, 3))
   for j in range(3):
      for k in range(3):
         b_23[j][k] = abs(B2[j][k] - B3[j][k])
   b_13 = np.zeros((3, 3))
   for j in range(3):
      for k in range(3):
         b_13[j][k] = abs(B1[j][k] - B3[j][k])      
   
   print("Distance between B1 and B2 : \n", b_12)
   print("Distance between B2 and B3 : \n", b_23)
   print("Distance between B1 and B3 : \n", b_13)
   # plt.legend()
   # plt.show()