import numpy as np
import matplotlib.pylab as plt

np.random.seed(65535)
a = np.random.normal(85,6,50).astype("int32")
a[np.where(a > 100)] = 100
b = np.random.normal(82,7,50).astype("int32")
b[np.where(b > 100)] = 100

print(a)
print()
print(b)
print()

print("With means of 82 & 85:")
from scipy.stats import ttest_ind
t,p = ttest_ind(a,b,equal_var=False)
print("(t=%0.5f, p=%0.5f)" % (t,p))

from scipy.stats import mannwhitneyu
u,p = mannwhitneyu(a,b)
print("(U=%0.5f, p=%0.5f)" % (u,p))

plt.boxplot((a,b))
plt.xlabel("Group")
plt.ylabel("Test score")
plt.savefig("hypothesis_box_plot.png", dpi=300)
plt.close()

h,x = np.histogram(a, bins=10)
plt.bar(x[:-1],h, width=0.4*(x[1]-x[0]), label='Group A')
h,y = np.histogram(b, bins=10)
plt.bar(y[:-1]+(x[1]-x[0])/2, h, width=0.4*(x[1]-x[0]), label='Group B')
plt.legend(loc='upper left')
plt.ylabel('Counts')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("hypothesis_bar_plot.png", dpi=300)
plt.close()

# CI for Welch's t-test
from scipy import stats

def CI(a, b, alpha=0.05):
    n1, n2 = len(a), len(b) 
    s1, s2 = np.std(a, ddof=1)**2, np.std(b, ddof=1)**2
    df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
    tc = stats.t.ppf(1 - alpha/2, df)
    lo = (a.mean()-b.mean()) - tc*np.sqrt(s1/n1 + s2/n2)
    hi = (a.mean()-b.mean()) + tc*np.sqrt(s1/n1 + s2/n2)
    return lo, hi

lo, hi = CI(a, b)
print("CI95 = (%0.5f, %0.5f)" % (lo,hi))

#  Cohen's d
def Cohen_d(a,b):
    s1 = np.std(a, ddof=1)**2
    s2 = np.std(b, ddof=1)**2
    return (a.mean() - b.mean()) / np.sqrt(0.5*(s1+s2))

print("Cohen's d = %0.5f" % Cohen_d(a,b))

#  change the means to be one step closer
np.random.seed(65535)
a = np.random.normal(85,6,50).astype("int32")
a[np.where(a > 100)] = 100
b = np.random.normal(83,7,50).astype("int32")
b[np.where(b > 100)] = 100

print("With means of 83 & 85:")
t,p = ttest_ind(a,b,equal_var=False)
print("(t=%0.5f, p=%0.5f)" % (t,p))
u,p = mannwhitneyu(a,b)
print("(U=%0.5f, p=%0.5f)" % (u,p))

#  means one step further apart
np.random.seed(65535)
a = np.random.normal(85,6,50).astype("int32")
a[np.where(a > 100)] = 100
b = np.random.normal(81,7,50).astype("int32")
b[np.where(b > 100)] = 100

print("With means of 81 & 85:")
t,p = ttest_ind(a,b,equal_var=False)
print("(t=%0.5f, p=%0.5f)" % (t,p))
u,p = mannwhitneyu(a,b)
print("(U=%0.5f, p=%0.5f)" % (u,p))

#  Effect of sample size
np.random.seed(65535)
pt = []
et = []
pm = []
em = []
M = 25
n = [20,40,60,80,100,120,140,160,180,200,250,300,350,400,450,500,750,1000]
for i in n:
    p = []
    t = []
    for j in range(M):
        a = np.random.normal(85,6,i).astype("int32")
        a[np.where(a > 100)] = 100
        b = np.random.normal(84,7,i).astype("int32")
        b[np.where(b > 100)] = 100
        t.append(ttest_ind(a,b,equal_var=False)[1])
        p.append(mannwhitneyu(a,b)[1])
    pt.append(np.array(t).mean())
    et.append(np.array(t).std(ddof=1)/np.sqrt(M))
    pm.append(np.array(p).mean())
    em.append(np.array(p).std(ddof=1)/np.sqrt(M))
    if (i==1000):
        print("n=1000 Cohen's d = %0.5f" % Cohen_d(a,b))
pt = np.array(pt)
pm = np.array(pm)
et = np.array(et)
em = np.array(em)
plt.errorbar(n,pt,et,marker='o',label='t-test')
plt.errorbar(n,pm,em,marker='s',label='Mann-Whitney U')
plt.xlabel('Sample size')
plt.ylabel("$p$-value")
plt.legend(loc="upper right")
plt.tight_layout(pad=0,w_pad=0,h_pad=0)
plt.savefig("hypothesis_pvalue_plot.png", dpi=300)

