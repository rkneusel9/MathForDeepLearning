import numpy as np
from sklearn import datasets
from scipy.spatial.distance import mahalanobis

bc = datasets.load_breast_cancer()
d = bc.data
l = bc.target
i = np.argsort(np.random.random(len(d)))
d = d[i]
l = l[i]
xtrn, ytrn = d[:400], l[:400]
xtst, ytst = d[400:], l[400:]

i = np.where(ytrn == 0)
m0 = xtrn[i].mean(axis=0)
i = np.where(ytrn == 1)
m1 = xtrn[i].mean(axis=0)
S = np.cov(xtrn, rowvar=False)
SI= np.linalg.inv(S)

def score(xtst, ytst, m, SI):
    nc = 0
    for i in range(len(ytst)):
        d = np.array([mahalanobis(xtst[i],m[0],SI),
                      mahalanobis(xtst[i],m[1],SI)])
        c = np.argmin(d)
        if (c == ytst[i]):
            nc += 1
    return nc / len(ytst)

mscore = score(xtst, ytst, [m0,m1], SI)
escore = score(xtst, ytst, [m0,m1], np.identity(30))
print("Mahalanobis score = %0.4f" % mscore)
print("Euclidean   score = %0.4f" % escore)

