#
#  file: mnist.py
#
#  Train and test the small 14x14 MNIST dataset.
#
#  RTK, 03-Feb-2021
#  Last update:  06-Feb-2021
#
################################################################

import numpy as np
from NN import *

#  Load, reshape, and scale the data
x_train = np.load("../dataset/train_images_small.npy")
x_test  = np.load("../dataset/test_images_small.npy")
y_train = np.load("../dataset/train_labels_vector.npy")
y_test  = np.load("../dataset/test_labels.npy")

x_train = x_train.reshape(x_train.shape[0], 1, 14*14)
x_train /= 255
x_test = x_test.reshape(x_test.shape[0], 1, 14*14)
x_test /= 255

#  Build the network using sigmoid activations
net = Network()
net.add(FullyConnectedLayer(14*14, 100))
net.add(ActivationLayer())
net.add(FullyConnectedLayer(100, 50))
net.add(ActivationLayer())
net.add(FullyConnectedLayer(50, 10))
net.add(ActivationLayer())

#  Loss and train
net.fit(x_train, y_train, minibatches=40000, learning_rate=1.0)

#  Build the confusion matrix using the test set predictions
out = net.predict(x_test)
cm = np.zeros((10,10), dtype="uint32")
for i in range(len(y_test)):
    cm[y_test[i],np.argmax(out[i])] += 1

#  Show the results
print()
print(np.array2string(cm))
print()
print("accuracy = %0.7f" % (np.diag(cm).sum() / cm.sum(),))
print()

