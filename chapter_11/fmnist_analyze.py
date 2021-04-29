import numpy as np
import matplotlib.pylab as plt
from scipy.stats import ttest_ind, mannwhitneyu

def Cohen_d(a,b):
    s1 = np.std(a, ddof=1)**2
    s2 = np.std(b, ddof=1)**2
    return (a.mean() - b.mean()) / np.sqrt(0.5*(s1+s2))

#  Load the MCC for repeated trainings
m_no = np.load("fmnist_no_momentum_runs.npy")
m_w = np.load("fmnist_w_momentum_runs.npy")

hn,xn = np.histogram(m_no, bins=5)
hw,xw = np.histogram(m_w, bins=5)
b = plt.bar(xn[:-1], hn, width=0.8*(xn[1]-xn[0]), hatch="/", color="#5f5f5f")
b = plt.bar(xw[:-1], hw, width=0.8*(xn[1]-xn[0]), hatch="\\", color="#7f7f7f")
plt.xlabel("MCC")
plt.ylabel("Count")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("fmnist_mcc_plot.png", dpi=300)
plt.show()

print()
print("no momentum: %0.5f +/- %0.5f" % (m_no.mean(), m_no.std(ddof=1)/np.sqrt(len(m_no))))
print("momentum   : %0.5f +/- %0.5f" % (m_w.mean(), m_w.std(ddof=1)/np.sqrt(len(m_w))))
print()
t,p = ttest_ind(m_w, m_no)
print("t-test momentum vs no (t,p): (%0.8f, %0.8f)" % (t,p))
U,p = mannwhitneyu(m_w, m_no)
print("Mann-Whitney U             : (%0.8f, %0.8f)" % (U,p))
print("Cohen's d                  : %0.5f" % Cohen_d(m_w, m_no))
print()

