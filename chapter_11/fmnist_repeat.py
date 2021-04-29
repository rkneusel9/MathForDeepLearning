#
#  file: fmnist_repeat.py
#
#  Train and test the small 14x14 FMNIST dataset.
#
#  RTK, 03-Feb-2021
#  Last update:  21-Feb-2021
#
################################################################

from sklearn.metrics import matthews_corrcoef
import numpy as np
from NNm import *

#  Load, reshape, and scale the data
x_train = np.load("../dataset/fmnist_train_images_small.npy")/255
x_test  = np.load("../dataset/fmnist_test_images_small.npy")/255
y_train = np.load("../dataset/fmnist_train_labels_vector.npy")
y_test  = np.load("../dataset/fmnist_test_labels.npy")

x_train = x_train.reshape(x_train.shape[0], 1, 14*14)
x_test = x_test.reshape(x_test.shape[0], 1, 14*14)

def train_test(x_train, x_test, y_train, y_test):
    #  Build the network using sigmoid activations
    net = Network(verbose=False)
    net.add(FullyConnectedLayer(14*14, 100, momentum=0.9))
    net.add(ActivationLayer())
    net.add(FullyConnectedLayer(100, 50, momentum=0.9))
    net.add(ActivationLayer())
    net.add(FullyConnectedLayer(50, 10, momentum=0.9))
    net.add(ActivationLayer())

    #  Loss and train
    net.fit(x_train, y_train, minibatches=10000, learning_rate=0.2)

    out = net.predict(x_test)
    pred = np.array(out)[:,0,:]
    return matthews_corrcoef(y_test, np.argmax(pred, axis=1))


M = 100
mcc = np.zeros(M)

for i in range(M):
    mcc[i] = train_test(x_train, x_test, y_train, y_test)
    print("%03d: MCC = %0.8f" % (i, mcc[i]), flush=True)

np.save("fmnist_repeat_mcc.npy", mcc)

print()
print("Overall MCC %0.6f +/- %0.6f" % (mcc.mean(), mcc.std(ddof=1)/np.sqrt(M)))
print()

