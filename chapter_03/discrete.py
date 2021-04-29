# discrete probability distributions
import numpy as np
import matplotlib.pylab as plt
from scipy.misc import face
from fldr import *
from fldrf import *

#  binomial
q = np.random.binomial(10, 0.7, 1000)
h = np.histogram(q, bins=q.max()-q.min()+1)[0]
h = h / h.sum()
x = np.arange(q.min(), q.max()+1)
plt.bar(x,h,width=0.8)
q = np.random.binomial(10, 0.3, 1000)
h = np.histogram(q, bins=q.max()-q.min()+1)[0]
h = h / h.sum()
x = np.arange(q.min(), q.max()+1)
plt.bar(x,h,width=0.8)
plt.show()

#  FLDR
im = face(True)
b = np.bincount(im.ravel(), minlength=256)
b = b / b.sum()
x = fldr_preprocess_float_c(list(b))
t = [fldr_sample(x) for i in range(500000)]
q = np.bincount(t, minlength=256)
q = q / q.sum()

plt.plot(b, color='k')
plt.plot(q, linestyle=(0, (1,1)), color='k')
plt.xlabel("Sample")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("fldr_samples.png", dpi=300)
plt.show()

