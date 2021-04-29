#
#  file: fmnist_momentum.py
#
#  Train and test the small 14x14 FMNIST dataset.
#
#  RTK, 03-Feb-2021
#  Last update:  19-Feb-2021
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

#  Build the network using sigmoid activations
net = Network(verbose=True)
net.add(FullyConnectedLayer(14*14, 100, momentum=0.9))
net.add(ActivationLayer())
net.add(FullyConnectedLayer(100, 50, momentum=0.9))
net.add(ActivationLayer())
net.add(FullyConnectedLayer(50, 10, momentum=0.9))
net.add(ActivationLayer())

#  Loss and train
net.fit(x_train, y_train, minibatches=40000, learning_rate=0.2)

#  Build the confusion matrix using the test set predictions
out = net.predict(x_test)
pred = np.array(out)[:,0,:]
cm = np.zeros((10,10), dtype="uint32")
for i in range(len(y_test)):
    cm[y_test[i],np.argmax(out[i])] += 1

#  Show the results
print()
print(np.array2string(cm))
print()
print("accuracy = %0.7f" % (np.diag(cm).sum() / cm.sum(),))
print("MCC = %0.7f" % matthews_corrcoef(y_test, np.argmax(pred, axis=1)))
print()

