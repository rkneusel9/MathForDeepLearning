#
#  file:  gd_1d.py
#
#  1D example of GD
#
#  RTK, 14-Feb-2021
#  Last update:  14-Feb-2021
#
################################################################

import sys
import os
import numpy as np
import matplotlib.pylab as plt

#  The function and its derivative
def f(x):
    return 6*x**2 - 12*x + 3

def d(x):
    return 12*x - 12


#  Show the function, derivative, and minimum
x = np.linspace(-1,3,1000)
y = f(x)
plt.plot(x,y,color='#1f77b4')
x = np.linspace(0,3,10)
z = d(x)
plt.plot(x,z,color='#ff7f0e')
plt.plot([-1,3],[0,0],linestyle=(0,(1,1)),color='k')
plt.plot([1,1],[-10,25],linestyle=(0,(1,1)),color='k')
plt.plot([1,1],[f(1),f(1)],marker='o',color='#1f77b4')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_1d_plot.png", dpi=300)
#plt.show()
plt.close()

#  Show a series of gradient descent steps
x = np.linspace(-1,3,1000)
plt.plot(x,f(x))

x = -0.9
eta = 0.03
for i in range(15):
    plt.plot(x, f(x), marker='o', color='r')
    x = x - eta * d(x)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_1d_steps.png", dpi=300)
#plt.show()
plt.close()
print("Minimum at (%0.6f, %0.6f)" % (x, f(x)))

#  Show oscillation if step size too large
x = np.linspace(0.75,1.25,1000)
plt.plot(x,f(x))
x = xold = 0.75
for i in range(14):
    plt.plot([xold,x], [f(xold),f(x)], marker='o', linestyle='dotted', color='r')
    xold = x
    x = x - 0.15 * d(x)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_1d_oscillating.png", dpi=300)
#plt.show()

