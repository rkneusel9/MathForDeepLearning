# plots of key continuous distributions
import numpy as np
import matplotlib.pylab as plt

N = 10000000
B = 100
x = np.arange(B)/B

#  uniform
t = np.random.random(N)
u = np.histogram(t, bins=B)[0]
u = u / u.sum()

#  normal
t = np.random.normal(0, 1, size=N)
n = np.histogram(t, bins=B)[0]
n = n / n.sum()

#  gamma
t = np.random.gamma(5.0, size=N)
g = np.histogram(t, bins=B)[0]
g = g / g.sum()

#  beta
t = np.random.beta(5,2, size=N)
b = np.histogram(t, bins=B)[0]
b = b / b.sum()

plt.plot(x,u,color='k',linestyle='solid')
plt.plot(x,n,color='k',linestyle='dotted')
plt.plot(x,g,color='k',linestyle='dashed')
plt.plot(x,b,color='k',linestyle='dashdot')
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("continuous.png", dpi=300)
plt.show()
plt.close()

#  central limit theorem
M = 10000
m = np.zeros(M)
for i in range(M):
    t = np.random.beta(5,2,size=M)
    m[i] = t.mean()
print("Mean of the means = %0.7f" % m.mean())

h,x = np.histogram(m, bins=B)
h = h / h.sum()
plt.bar(x[:-1]+0.5*(x[1]-x[0]), h, width=0.8*(x[1]-x[0]))
plt.xlabel("Mean")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("central_limit.png", dpi=300)
plt.show()
plt.close()

from fldrf import fldr_preprocess_float_c
from fldr import fldr_sample

z = fldr_preprocess_float_c([0.1,0.6,0.1,0.1,0.1])
m = np.zeros(M)
for i in range(M):
    t = np.array([fldr_sample(z) for i in range(M)])
    m[i] = t.mean()
print("Mean of the means = %0.7f" % m.mean())

h,x = np.histogram(m, bins=B)
h = h / h.sum()
plt.bar(x[:-1]+0.5*(x[1]-x[0]), h, width=0.8*(x[1]-x[0]))
plt.xlabel("Mean")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("central_limit_fldr.png", dpi=300)
plt.show()
plt.close()

t = np.array([fldr_sample(z) for i in range(M)])
h = np.bincount(t)
h = h / h.sum()
plt.bar(np.arange(5),h, width=0.8)
plt.xlabel("Value")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("pmf_fldr.png", dpi=300)
plt.show()
plt.close()

#  Law of large numbers
m = []
for n in np.linspace(1,8,30):
    t = np.random.normal(1,1,size=int(10**n))
    m.append(t.mean())

plt.plot(np.linspace(1,8,30), m)
plt.plot([1,8],[1,1], linestyle="--", color='k')
plt.xlabel("Exponent $10^n$")
plt.ylabel("Single sample mean")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig("large_numbers.png", dpi=300)
plt.show()


