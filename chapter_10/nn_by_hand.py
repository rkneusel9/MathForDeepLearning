#
#  file:  nn_by_hand.py
#
#  Implement a simple feedforward neural network with
#  backprop and gradient descent.
#
#  RTK, 02-Feb-2021
#  Last update:  02-Feb-2021
#
################################################################

import numpy as np
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

    return xtrn, ytrn, xtst, ytst


################################################################
#  sigmoid
#
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


################################################################
#  Forward
#
def Forward(net, x):
    """Pass the data through the network"""

    out = np.zeros(x.shape[0])

    for k in range(x.shape[0]):
        z0 = net["w0"]*x[k,0] + net["w2"]*x[k,1] + net["b0"]
        a0 = sigmoid(z0)
        z1 = net["w1"]*x[k,0] + net["w3"]*x[k,1] + net["b1"]
        a1 = sigmoid(z1)
        out[k] = net["w4"]*a0 + net["w5"]*a1 + net["b2"]

    return out


################################################################
#  Evaluate
#
def Evaluate(net, x, y):
    """Evaluate the network"""

    out = Forward(net, x)
    tn = fp = fn = tp = 0
    pred = []
    
    for i in range(len(y)):
        c = 0 if (out[i] < 0.5) else 1
        pred.append(c)
        if (c == 0) and (y[i] == 0):
            tn += 1
        elif (c == 0) and (y[i] == 1):
            fn += 1
        elif (c == 1) and (y[i] == 0):
            fp += 1
        else:
            tp += 1
    
    return tn,fp,fn,tp,pred



################################################################
#  GradientDescent
#
def GradientDescent(net, x, y, epochs, eta):
    """Perform gradient descent"""

    for e in range(epochs):
        #  Pass over training set accumulating deltas
        dw0 = dw1 = dw2 = dw3 = dw4 = dw5 = db0 = db1 = db2 = 0.0

        for k in range(len(y)):
            #  Forward pass
            z0 = net["w0"]*x[k,0] + net["w2"]*x[k,1] + net["b0"]
            a0 = sigmoid(z0)
            z1 = net["w1"]*x[k,0] + net["w3"]*x[k,1] + net["b1"]
            a1 = sigmoid(z1)
            a2 = net["w4"]*a0 + net["w5"]*a1 + net["b2"]

            #  Backward pass
            db2 += a2 - y[k]
            dw4 += (a2 - y[k]) * a0
            dw5 += (a2 - y[k]) * a1
            db1 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1)
            dw1 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1) * x[k,0]
            dw3 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1) * x[k,1]
            db0 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0)
            dw0 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0) * x[k,0]
            dw2 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0) * x[k,1]

        #  Use average deltas to update the network
        m = len(y)
        net["b2"] = net["b2"] - eta * db2 / m
        net["w4"] = net["w4"] - eta * dw4 / m
        net["w5"] = net["w5"] - eta * dw5 / m
        net["b1"] = net["b1"] - eta * db1 / m
        net["w1"] = net["w1"] - eta * dw1 / m
        net["w3"] = net["w3"] - eta * dw3 / m
        net["b0"] = net["b0"] - eta * db0 / m
        net["w0"] = net["w0"] - eta * dw0 / m
        net["w2"] = net["w2"] - eta * dw2 / m

    #  Training done, return the updated network
    return net


################################################################
#  main
#
def main():
    """Build and train a simple neural network"""

    epochs = 1000  # training epochs
    eta = 0.1      # learning rate

    #  Get the train/test data
    xtrn, ytrn, xtst, ytst = BuildDataset()

    #  Initialize the network
    net = {}
    net["b2"] = 0.0
    net["b1"] = 0.0
    net["b0"] = 0.0
    net["w5"] = 0.0001*(np.random.random() - 0.5)
    net["w4"] = 0.0001*(np.random.random() - 0.5)
    net["w3"] = 0.0001*(np.random.random() - 0.5)
    net["w2"] = 0.0001*(np.random.random() - 0.5)
    net["w1"] = 0.0001*(np.random.random() - 0.5)
    net["w0"] = 0.0001*(np.random.random() - 0.5)

    #  Do a forward pass to get initial performance
    tn0,fp0,fn0,tp0,pred0 = Evaluate(net, xtst, ytst)

    #  Gradient descent
    net = GradientDescent(net, xtrn, ytrn, epochs, eta)

    #  Final model performance
    tn,fp,fn,tp,pred = Evaluate(net, xtst, ytst)

    #  Summarize performance
    print()
    print("Training for %d epochs, learning rate %0.5f" % (epochs, eta))
    print()
    print("Before training:")
    print("    TN:%3d  FP:%3d" % (tn0, fp0))
    print("    FN:%3d  TP:%3d" % (fn0, tp0))
    print()
    print("After training:")
    print("    TN:%3d  FP:%3d" % (tn, fp))
    print("    FN:%3d  TP:%3d" % (fn, tp))
    print()


if (__name__ == "__main__"):
    main()


