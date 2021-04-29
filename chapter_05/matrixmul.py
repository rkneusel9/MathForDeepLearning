#  O(n^3) matrix multiplication

import time
import numpy as np

def matrixmul(A,B):
    I,K = A.shape
    J = B.shape[1]
    C = np.zeros((I,J), dtype=A.dtype)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                C[i,j] += A[i,k]*B[k,j]
    return C
    
A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
B = np.array([[1,2],[3,4],[5,6]])
N = 100000

s = time.time()
for i in range(N):
    C = np.matmul(A,B)
e = time.time()
print("np.matmul: %0.6f" % (e-s,))

s = time.time()
for i in range(N):
    C = matrixmul(A,B)
e = time.time()
print("matrixmul: %0.6f" % (e-s,))

