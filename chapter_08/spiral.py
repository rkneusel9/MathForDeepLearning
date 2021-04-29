import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

t = np.linspace(0,50,1000)
x = t*np.cos(t)
y = t*np.sin(t)
z = t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='k')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("spiral.png", dpi=300)
plt.show()

