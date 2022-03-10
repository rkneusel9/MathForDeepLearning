#
#  file:  gd_multiple.py
#
#  2D example of gradient descent for a function
#  with more than one minimum
#
#  RTK, 14-Feb-2021
#  Last update:  14-Feb-2021
#
################################################################

import numpy as np
import matplotlib.pylab as plt

#  Function and partial derivatives
def f(x,y):
    return -2*np.exp(-0.5*((x+1)**2+(y-1)**2)) +  \
           -np.exp(-0.5*((x-1)**2+(y+1)**2))

def dx(x,y):
    return 2*(x+1)*np.exp(-0.5*((x+1)**2+(y-1)**2)) +  \
           (x-1)*np.exp(-0.5*((x-1)**2+(y+1)**2))

def dy(x,y):
    return (y+1)*np.exp(-0.5*((x-1)**2+(y+1)**2)) +  \
           2*(y-1)*np.exp(-0.5*((x+1)**2+(y-1)**2))

#  Gradient descent steps
N = 100
x,y = np.meshgrid(np.linspace(-2,2,N), np.linspace(-2,2,N))
z = f(x,y)
plt.contourf(x,y,z,10, cmap="Greys")
plt.contour(x,y,z,10, colors='k', linewidths=1)
plt.plot([0,0],[-2,2],color='k',linewidth=1)
plt.plot([-2,2],[0,0],color='k',linewidth=1)

eta = 0.4

x = xold = -1.5
y = yold = 1.2
for i in range(9):
    plt.plot([xold,x],[yold,y], marker='o', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - eta * dx(x,y)
    y = y - eta * dy(x,y)

x = xold = 1.5
y = yold = -1.8
for i in range(9):
    plt.plot([xold,x],[yold,y], marker='s', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - eta * dx(x,y)
    y = y - eta * dy(x,y)

x = xold = 0.0
y = yold = 0.0
for i in range(20):
    plt.plot([xold,x],[yold,y], marker='+', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - eta * dx(x,y)
    y = y - eta * dy(x,y)

x = xold = 0.7
y = yold = -0.2
for i in range(20):
    plt.plot([xold,x],[yold,y], marker='>', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - eta * dx(x,y)
    y = y - eta * dy(x,y)

x = xold = 1.5
y = yold = 1.5
for i in range(30):
    plt.plot([xold,x],[yold,y], marker='*', linestyle='dotted', color='k')
    xold = x
    yold = y
    x = x - eta * dx(x,y)
    y = y - eta * dy(x,y)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_multiple_steps.png", dpi=300)
plt.show()
plt.close()

