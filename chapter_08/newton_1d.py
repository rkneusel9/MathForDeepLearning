#  Newton's method in 1D
import numpy as np

def f(x):
    return 2.0 - x*x

def d(x):
    return -2.0*x

x = 1.0

for i in range(5):
    x = x - f(x)/d(x)
    print("%2d: %0.16f" % (i+1,x))

print()
print("NumPy says sqrt(2) = %0.16f for a deviation of %0.16f" % (np.sqrt(2), np.abs(np.sqrt(2)-x)))
print()

