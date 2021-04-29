#
#  file: c07_figure.py
#
#  Plot of x^2+xy+y^2 and gradient field.
#
#  RTK, 25-Mar-2020
#  Last update:  26-Mar-2020
#
################################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

#  Function plot
x = np.linspace(-1.0,1.0,50)
y = np.linspace(-1.0,1.0,50)
xx = []
yy = []
zz = []

for i in range(50):
    for j in range(50):
        xx.append(x[i])
        yy.append(y[j])
        zz.append(x[i]*x[i]+x[i]*y[j]+y[j]*y[j])
x = np.array(xx)
y = np.array(yy)
z = np.array(zz)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, marker='.', s=2, color='b')
ax.view_init(30, 50)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.draw()
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("c05fig03a.png", dpi=300)
ax.view_init(30,20)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.draw()
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("c05fig03b.png", dpi=300)
plt.close()

# Quiver plot - 2D
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(-1.0,1.0,20)
y = np.linspace(-1.0,1.0,20)
xv, yv = np.meshgrid(x, y, indexing='ij', sparse=False)
dx = 2*xv + yv
dy = 2*yv + xv
ax.quiver(xv, yv, dx, dy, color='b')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.axis('equal')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("c05fig03c.png", dpi=300)
plt.close()


