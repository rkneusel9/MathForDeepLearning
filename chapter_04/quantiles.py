#
#  Use a synthetic exam dataset to illustrate quantiles
#
#  RTK, 03-Jul-2020
#  Last update:  03-Jul-2020
#
################################################################

import numpy as np
import matplotlib.pylab as plt

d = np.load("exams.npy")
p = d[:,0].astype("uint32")
q = np.quantile(p, [0.0, 0.25, 0.5, 0.75, 1.0])

print()
print("Quartiles: ", q)
print()
print("Counts by quartile:")
print("    %d" % ((q[0] <= p) & (p < q[1])).sum())
print("    %d" % ((q[1] <= p) & (p < q[2])).sum())
print("    %d" % ((q[2] <= p) & (p < q[3])).sum())
print("    %d" % ((q[3] <= p) & (p < q[4])).sum())
print()

h = np.bincount(p, minlength=100)
x = np.arange(101)
plt.bar(x,h, width=0.8*(x[1]-x[0]))
n = 1.1*h.max()
plt.plot([q[1],q[1]],[0,n], linewidth=3, color='k')
plt.plot([q[2],q[2]],[0,n], linewidth=3, color='k')
plt.plot([q[3],q[3]],[0,n], linewidth=3, color='k')
plt.xlim((p.min()-1,p.max()+1))
plt.ylabel("Count")
plt.tight_layout(pad=0,w_pad=0,h_pad=0)
plt.savefig("quantiles_plot.png", dpi=300)
#plt.show()
plt.close()

# box plot
plt.boxplot(d)
plt.xlabel("Test")
plt.ylabel("Scores")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("box_plot.png", dpi=300)
#plt.show()
plt.close()

plt.boxplot(p)
plt.ylabel("Scores")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("box_plot_1.png", dpi=300)
plt.show()
plt.close()

