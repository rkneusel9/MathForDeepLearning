import numpy as np
import matplotlib.pylab as plt

np.random.seed(8675309)

N = 100
x = np.linspace(0,1,N) + (np.random.random(N)-0.5)
y = np.random.random(N)*x
z = -0.1*np.random.random(N)*x

plt.plot(np.linspace(0,1,N),x,color='r')
plt.plot(np.linspace(0,1,N),y,color='g')
plt.plot(np.linspace(0,1,N),z,color='b')
plt.plot(np.linspace(0,1,N)[::5],x[::5],color='r',marker='o',linestyle='none',label='X')
plt.plot(np.linspace(0,1,N)[::5],y[::5],color='g',marker='s',linestyle='none',label='Y')
plt.plot(np.linspace(0,1,N)[::5],z[::5],color='b',marker='*',linestyle='none',label='Z')
plt.legend(loc="upper left")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("correlation_xyz_plot.png", dpi=300)
plt.close()

plt.plot(x,y,marker='o',linestyle='none',color='r',label="X,Y")
plt.plot(x,z,marker='s',linestyle='none',color='g',label="X,Z")
plt.plot(y,z,marker='*',linestyle='none',color='b',label="Y,Z")
plt.legend(loc="upper left")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("correlation_scatter_plot.png", dpi=300)
plt.close()


from scipy.stats import pearsonr, spearmanr

print("Pearson(x,y) :", pearsonr(x,y)[0])
print("Spearman(x,y):", spearmanr(x,y)[0])
print()
print("Pearson(x,z) :", pearsonr(x,z)[0])
print("Spearman(x,z):", spearmanr(x,z)[0])
print()
print("Pearson(y,z) :", pearsonr(y,z)[0])
print("Spearman(y,z):", spearmanr(y,z)[0])
print()

def pearson(x,y):
    exy = (x*y).mean()
    ex = x.mean()
    ey = y.mean()
    exx = (x*x).mean()
    ex2 = x.mean()**2
    eyy = (y*y).mean()
    ey2 = y.mean()**2
    return (exy - ex*ey)/(np.sqrt(exx-ex2)*np.sqrt(eyy-ey2))

print("pearson(x,y):", pearson(x,y))
print("pearson(x,z):", pearson(x,z))
print("pearson(y,z):", pearson(y,z))
print()

d = np.vstack((x,y,z))
print(np.corrcoef(d))
print()

from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')
a = china[230,:,1].astype("float64")
b = china[231,:,1].astype("float64")
c = china[400,:,1].astype("float64")
d = np.random.random(640)
print("china(a,b): ", pearson(a,b))
print("china(a,c): ", pearson(a,c))
print("china(a,d): ", pearson(a,d))
print()

#  spearman
def spearman(x,y):
    n = len(x)
    t = x[np.argsort(x)]
    rx = []
    for i in range(n):
        rx.append(np.where(x[i] == t)[0][0])
    rx = np.array(rx, dtype="float64")
    t = y[np.argsort(y)]
    ry = []
    for i in range(n):
        ry.append(np.where(y[i] == t)[0][0])
    ry = np.array(ry, dtype="float64")
    d = rx - ry
    return 1.0 - (6.0/(n*(n*n-1)))*(d**2).sum()

print(spearman(x,y), spearmanr(x,y)[0])
print(spearman(x,z), spearmanr(x,z)[0])
print(spearman(y,z), spearmanr(y,z)[0])
print()

a = np.linspace(-20,20,1000)
b = 1.0 / (1.0 + np.exp(-a))
print(pearson(a,b))
print(spearman(a,b))



