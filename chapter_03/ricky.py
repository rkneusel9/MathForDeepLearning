import numpy as np
import matplotlib.pylab as plt
import scipy.misc
from PIL import Image

im = scipy.misc.face(True)[:512,512:]
Image.fromarray(im).save("ricky.png")
hr,xr = np.histogram(im, bins=256)
hr = hr/hr.sum()
im = scipy.misc.ascent().astype("uint8")
Image.fromarray(im).save("ascent.png")
ha,xa = np.histogram(im, bins=256)
ha = ha/ha.sum()
plt.plot(xr[:-1],hr, color='k', label="Face")
plt.plot(xa[:-1],ha, linestyle=(0,(1,1)), color='k', label="Ascent")
plt.legend(loc="upper right")
plt.xlabel("Gray level")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("ricky_probability.png", dpi=300)
plt.show()
plt.close()


