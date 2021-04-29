#
#  file:  gd_1d_momentum.py
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

m = ['o','s','>','<','*','+','p','h','P','D']
x = np.linspace(0.75,1.25,1000)
plt.plot(x,f(x))
x = xold = 0.75
eta = 0.09
mu = 0.8
v = 0.0
for i in range(10):
    plt.plot([xold,x], [f(xold),f(x)], marker=m[i], linestyle='dotted', color='r')
    xold = x
    v = mu*v - eta * d(x)
    x = x + v
for i in range(40):
    v = mu*v - eta * d(x)
    x = x + v
plt.plot(x,f(x),marker='X', color='k')

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_1d_momentum.png", dpi=300)
plt.show()

