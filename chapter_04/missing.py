#  Missing data example
import numpy as np
import matplotlib.pylab as plt

N = 1000
np.random.seed(73939133)
x = np.zeros((N,4))
x[:,0] = 5*np.random.random(N)
x[:,1] = np.random.normal(10,1,size=N)
x[:,2] = 3*np.random.beta(5,2,N)
x[:,3] = 0.3*np.random.lognormal(size=N)

plt.boxplot(x)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("missing_box_plot.png", dpi=300)
plt.close()

#  Make 5% of the values NaN
i = np.random.randint(0,N, size=int(0.05*N))
x[i,0] = np.nan
i = np.random.randint(0,N, size=int(0.05*N))
x[i,1] = np.nan
i = np.random.randint(0,N, size=int(0.05*N))
x[i,2] = np.nan
i = np.random.randint(0,N, size=int(0.05*N))
x[i,3] = np.nan

#  Do we have NaNs in feature 2?
if (np.isnan(x[:,2]).sum() != 0):
    print("NaNs present")
    i = np.where(np.isnan(x[:,2]) == False)
    z = x[i,2]
    mn,md,s = z.mean(), np.median(z), z.std(ddof=1)
    hh,xx = np.histogram(z, bins=40)
    plt.bar(xx[:-1],hh, width=0.8*(xx[1]-xx[0]))
    plt.xlabel("x")
    plt.ylabel("Count")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig("missing_feature_2_plot.png", dpi=300)
    plt.close()

    i = np.where(np.isnan(x[:,2]) == True)
    x[i,2] = md  # replace w/median
    
    print("non-NaN mean, std = ", z.mean(), z.std(ddof=1))
    print("updated mean, std = ", x[:,2].mean(), x[:,2].std(ddof=1))

    hh,xx = np.histogram(x[:,2], bins=40)
    plt.bar(xx[:-1],hh, width=0.8*(xx[1]-xx[0]))
    plt.xlabel("x")
    plt.ylabel("Count")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig("missing_feature_2_updated_plot.png", dpi=300)
    plt.close()

#  Do the same to the others
i = np.where(np.isnan(x[:,0]) == False)
m = np.median(x[i,0])
i = np.where(np.isnan(x[:,0]) == True)
x[i,0] = m

i = np.where(np.isnan(x[:,1]) == False)
m = np.median(x[i,1])
i = np.where(np.isnan(x[:,1]) == True)
x[i,1] = m

i = np.where(np.isnan(x[:,3]) == False)
m = np.median(x[i,3])
i = np.where(np.isnan(x[:,3]) == True)
x[i,3] = m

plt.boxplot(x)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("missing_updated_box_plot.png", dpi=300)
plt.close()



