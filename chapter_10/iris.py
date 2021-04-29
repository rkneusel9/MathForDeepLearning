#
#  file: iris.py
#
#  Train and test the 2-feature iris dataset
#
#  RTK, 06-Feb-2021
#  Last update:  06-Feb-2021
#
################################################################

import numpy as np
from NN import *
from sklearn.datasets import load_iris

def BuildDataset():
    """Create the dataset"""

    #  Get the dataset keeping the first two features
    iris = load_iris()
    x = iris["data"][:,:2]
    y = iris["target"]

    #  Standardize and keep only classes 0 and 1
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    i0 = np.where(y == 0)[0]
    i1 = np.where(y == 1)[0]
    x = np.vstack((x[i0],x[i1]))

    #  Train and test data
    xtrn = np.vstack((x[:35],x[50:85]))
    ytrn = np.array([0]*35 + [1]*35)
    xtst = np.vstack((x[35:50],x[85:]))
    ytst = np.array([0]*15+[1]*15)
    
    idx = np.argsort(np.random.random(70))
    xtrn = xtrn[idx]
    ytrn = ytrn[idx]
    idx = np.argsort(np.random.random(30))
    xtst = xtst[idx]
    ytst = ytst[idx]

    y_train = np.zeros((len(ytrn),2))
    for i in range(len(ytrn)):
        if (ytrn[i] == 1):
            y_train[i,:] = [0,1]
        else:
            y_train[i,:] = [1,0]

    y_test = np.zeros((len(ytst),2))
    for i in range(len(ytst)):
        if (ytst[i] == 1):
            y_test[i,:] = [0,1]
        else:
            y_test[i,:] = [1,0]

    return (xtrn.reshape((xtrn.shape[0],1,2)), y_train,
            xtst.reshape((xtst.shape[0],1,2)), y_test)


def main():
    """Train a model"""

    x_train, y_train, x_test, y_test = BuildDataset()

    #  Build the network using sigmoid activations
    net = Network()
    net.add(FullyConnectedLayer(2,2))
    net.add(ActivationLayer())
    net.add(FullyConnectedLayer(2,2))

    #  Loss and train
    net.fit(x_train, y_train, minibatches=4000, learning_rate=0.1, batch_size=len(y_train))

    #  Build the confusion matrix using the test set predictions
    out = net.predict(x_test)
    cm = np.zeros((2,2), dtype="uint32")
    for i in range(len(y_test)):
        cm[np.argmax(y_test[i]),np.argmax(out[i])] += 1

    #  Show the results
    print()
    print(np.array2string(cm))
    print()
    print("accuracy = %0.7f" % (np.diag(cm).sum() / cm.sum(),))
    print()


if (__name__ == "__main__"):
    main()

