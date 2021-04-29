#
#  file:  gd_nesterov.py
#
#  2D example of gradient descent for a function
#  with more than one minimum and Nesterov momentum
#
#  RTK, 14-Feb-2021
#  Last update:  21-Feb-2021
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

def gd(x,y, eta,mu, steps, marker):
    xold = x
    yold = y
    vx = vy = 0.0
    for i in range(steps):
        plt.plot([xold,x],[yold,y], marker=marker, linestyle='dotted', color='k')
        xold = x
        yold = y
        vx = mu*vx - eta * dx(x+mu*vx,y)
        vy = mu*vy - eta * dy(x,y+mu*vy)
        x = x + vx
        y = y + vy

    return x,y

#gd(-1.5, 1.2,20, 'o')
#gd( 1.5,-1.8,40, 's')
#gd( 0.0, 0.0,30, '<')
print("(x,y) = (%0.8f, %0.8f)" % gd( 0.7,-0.2, 0.1,  0.9, 25, '>'))
print("(x,y) = (%0.8f, %0.8f)" % gd( 1.5, 1.5, 0.02, 0.9, 90, '*'))

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gd_nesterov_steps.png", dpi=300)
plt.show()
plt.close()

