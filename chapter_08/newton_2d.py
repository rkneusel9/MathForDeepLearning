# 2D Newton's method
import numpy as np

def f(x):
    x0,x1 = x[0,0],x[1,0]
    return np.array([[4*x0-2*x0*x1],[2*x1+x0*x1-2*x1**2]])

def JI(x):
    x0,x1 = x[0,0],x[1,0]
    d = (4-2*x1)*(2-x0-4*x1)+2*x0*x1
    return (1/d)*np.array([[2-x0-4*x1,2*x0],[-x1,4-2*x0]])

x0 = float(input("x0: "))
x1 = float(input("x1: "))
x = np.array([[x0],[x1]])

N = 20
for i in range(N):
    x = x - JI(x) @ f(x)
    if (i > (N-10)):
        print("%4d: (%0.8f, %0.8f)" % (i, x[0,0],x[1,0]))

