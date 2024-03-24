import numpy as np
import torch

def test_numpy_dot(matrix1, matrix2):

    # Compute dot product using NumPy
    result_numpy = np.dot(matrix1, matrix2)
    print("NumPy dot product result:")
    print(result_numpy.shape)
    print(result_numpy)

def test_torch_bmm(matrix1, matrix2):
    # Generate random tensors
    tensor1 = torch.tensor(matrix1).unsqueeze(0)
    tensor2 = torch.tensor(matrix2).unsqueeze(0)

    # Compute batch matrix multiplication using PyTorch
    result_torch = torch.bmm(tensor1, tensor2)
    print("\nPyTorch bmm result:")
    print(result_torch.shape)
    print(result_torch)

if __name__ == "__main__":
    # Generate random matrices
    matrix1 = np.random.rand(2, 3)
    matrix2 = np.random.rand(3, 4)

    print("Matrix 1:", matrix1)
    print("Matrix 2:", matrix2)

    test_numpy_dot(matrix1, matrix2)
    test_torch_bmm(matrix1, matrix2)