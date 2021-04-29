import numpy as np
from scipy.special import rel_entr
import matplotlib.pylab as plt

N = 1000000
p = np.random.randint(0,13,size=N)
p = np.bincount(p)
p = p / p.sum()
q = np.random.binomial(12,0.9,size=N)
q = np.bincount(q)
q = q / q.sum()
w = np.random.binomial(12,0.4,size=N)
w = np.bincount(w)
w = w / w.sum()
print(rel_entr(q,p).sum())
print(rel_entr(w,p).sum())
plt.bar(np.arange(13),p,0.333,hatch="///",edgecolor='k')
plt.bar(np.arange(13)+0.333,q,0.333,hatch="---",edgecolor='k')
plt.bar(np.arange(13)+0.666,w,0.333,hatch="\\\\",edgecolor='k')
plt.xlabel("Value")
plt.ylabel("Proportion")
plt.tight_layout(pad=0,h_pad=0,w_pad=0)
plt.savefig("kl_divergence.png", dpi=300)
plt.show()

