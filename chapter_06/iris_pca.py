import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_iris
from scipy.linalg import svd as SVD

iris = load_iris().data.copy()
labels = load_iris().target.copy()
m = iris.mean(axis=0)
s = iris.std(axis=0)
ir = iris - m
cv = np.cov(ir, rowvar=False)
val, vec = np.linalg.eig(cv)
val = np.abs(val)
idx = np.argsort(val)[::-1]
ex = val[idx] / val.sum()
print("fraction explained: ", ex)
w = np.vstack((vec[:,idx[0]],vec[:,idx[1]]))
d = np.zeros((ir.shape[0],2))
for i in range(ir.shape[0]):
    d[i,:] = np.dot(w,ir[i])

markers = np.array(["o","s","+"])[labels]
for i in range(len(labels)):
    plt.plot(d[i,0], d[i,1], marker=markers[i], color='k', linestyle='none')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("iris_pca.png", dpi=300)
plt.close()

pca = PCA(n_components=2)
pca.fit(ir)
dd = pca.fit_transform(ir)
for i in range(len(labels)):
    plt.plot(dd[i,0], dd[i,1], marker=markers[i], color='k', linestyle='none')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("iris_pca_sklearn.png", dpi=300)
plt.close()

svd = TruncatedSVD(n_components=2)
svd.fit(ir)
s = svd.fit_transform(ir)
for i in range(len(labels)):
    plt.plot(s[i,0], s[i,1], marker=markers[i], color='k', linestyle='none')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("iris_pca_sklearn_svd.png", dpi=300)
plt.close()

#  truncate manually - exact result as sklearn PCA
n_elements = 2
u,s,vt = SVD(ir)
S = np.zeros((ir.shape[0], ir.shape[1]))
for i in range(4):
    S[i,i] = s[i]
S = S[:, :n_elements]
T = u @ S
for i in range(len(labels)):
    plt.plot(T[i,0], T[i,1], marker=markers[i], color='k', linestyle='none')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("iris_pca_truncated_svd.png", dpi=300)


