#
#  file:  numpy_matmul.py
#
#  NumPy matrix multiplication examples
#
#  RTK, 12-Apr-2020
#  Last update:  12-Apr-2020
#
################################################################

import numpy as np

def dot(a,b):
    try:
        return np.dot(a,b)
    except:
        return "fails"

def matmul(a,b):
    try:
        return np.matmul(a,b)
    except:
        return "fails"

#  the different vectors and matrices
a1 = np.array([1,2,3])
ar = a1.reshape((1,3))
ac = a1.reshape((3,1))
b1 = np.array([1,2,3])
br = b1.reshape((1,3))
bc = b1.reshape((3,1))
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[3,2,1]])

print()
print("np.dot examples:")
print("dot(a1,b1):"); print(dot(a1,b1))
print("dot(a1,br):"); print(dot(a1,br))
print("dot(a1,bc):"); print(dot(a1,bc))
print("dot(ar,b1):"); print(dot(ar,b1))
print("dot(ar,br):"); print(dot(ar,br))
print("dot(ar,bc):"); print(dot(ar,bc))
print("dot(ac,b1):"); print(dot(ac,b1))
print("dot(ac,br):"); print(dot(ac,br))
print("dot(ac,bc):"); print(dot(ac,bc))
print("dot(A,a1):"); print(dot(A,a1))
print("dot(A,ar):"); print(dot(A,ar))
print("dot(A,ac):"); print(dot(A,ac))
print("dot(a1,A):"); print(dot(a1,A))
print("dot(ar,A):"); print(dot(ar,A))
print("dot(ac,A):"); print(dot(ac,A))
print("dot(A,B):"); print(dot(A,B))
print()

print()
print("np.matmul examples:")
print("matmul(a1,b1):"); print(matmul(a1,b1))
print("matmul(a1,br):"); print(matmul(a1,br))
print("matmul(a1,bc):"); print(matmul(a1,bc))
print("matmul(ar,b1):"); print(matmul(ar,b1))
print("matmul(ar,br):"); print(matmul(ar,br))
print("matmul(ar,bc):"); print(matmul(ar,bc))
print("matmul(ac,b1):"); print(matmul(ac,b1))
print("matmul(ac,br):"); print(matmul(ac,br))
print("matmul(ac,bc):"); print(matmul(ac,bc))
print("matmul(A,a1):"); print(matmul(A,a1))
print("matmul(A,ar):"); print(matmul(A,ar))
print("matmul(A,ac):"); print(matmul(A,ac))
print("matmul(a1,A):"); print(matmul(a1,A))
print("matmul(ar,A):"); print(matmul(ar,A))
print("matmul(ac,A):"); print(matmul(ac,A))
print("matmul(A,B):"); print(matmul(A,B))
print()

