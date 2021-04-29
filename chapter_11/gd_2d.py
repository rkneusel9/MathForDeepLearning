#
#  file:  gd_2d.py
#
#  2D example of gradient descent
#
#  RTK, 14-Feb-2021
#  Last update:  14-Feb-2021
#
################################################################

import numpy as np
import matplotlib.pylab as plt

#  Function and partial derivatives
def f(x,y):
    return 6*x**2 + 9*y**2 - 12*x - 14*y + 3

def dx(x):
    return 12*x - 12

def dy(y):
    return 18*y - 14

#  Gradient descent steps
N = 100
x,y = np.meshgrid(np.linspace(-1,3,N), np.linspace(-1,3,N))
z = f(x,y)
plt.contourf(x,y,z,10, cmap="Greys")
plt.contour(x,y,z,10, colors='k', linewidths=1)
plt.plot([0,0],[-1,3],color='k',linewidth=1)
plt.plot([-1,3],[0,0],color='k',linewidth=1)
plt.plot(1,0.7777778,color='k',marker='+')

x = xold = -0.5
y = yold = 2.9
for i in range(12):
    plt.plot([xold,x],[yold,y], marker='o', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - 0.02 * dx(x)
    y = y - 0.02 * dy(y)

x = xold = 1.5
y = yold = -0.8
for i in range(12):
    plt.plot([xold,x],[yold,y], marker='s', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - 0.02 * dx(x)
    y = y - 0.02 * dy(y)

x = xold = 2.7
y = yold = 2.3
for i in range(12):
    plt.plot([xold,x],[yold,y], marker='<', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - 0.02 * dx(x)
    y = y - 0.02 * dy(y)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_2d_steps.png", dpi=300)
plt.show()
plt.close()

#  New function and partial derivatives
def f(x,y):
    return 6*x**2 + 40*y**2 - 12*x - 30*y + 3

def dx(x):
    return 12*x - 12

def dy(y):
    return 80*y - 30

#  Large stepsize
N = 100
x,y = np.meshgrid(np.linspace(-1,3,N), np.linspace(-1,3,N))
z = f(x,y)
plt.contourf(x,y,z,10, cmap="Greys")
plt.contour(x,y,z,10, colors='k', linewidths=1)
plt.plot([0,0],[-1,3],color='k',linewidth=1)
plt.plot([-1,3],[0,0],color='k',linewidth=1)
plt.plot(1,0.375,color='k',marker='+')

x = xold = -0.5
y = yold = 2.3
for i in range(14):
    plt.plot([xold,x],[yold,y], marker='o', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - 0.02 * dx(x)
    y = y - 0.02 * dy(y)

x = xold = 2.3
y = yold = 2.3
for i in range(14):
    plt.plot([xold,x],[yold,y], marker='s', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - 0.01 * dx(x)
    y = y - 0.01 * dy(y)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_2d_oscillating.png", dpi=300)
plt.show()
plt.close()

