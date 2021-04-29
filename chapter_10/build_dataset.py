#
#  file:  build_dataset.py
#
#  Build the small MNIST dataset.
#
#  RTK, 03-Feb-2021
#  Last update:  03-Feb-2021
#
################################################################

import cv2
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
ytrn = np_utils.to_categorical(y_train)

np.save("../dataset/train_images_full.npy", x_train)
np.save("../dataset/test_images_full.npy", x_test)
np.save("../dataset/train_labels_vector.npy", ytrn)
np.save("../dataset/train_labels.npy", y_train)
np.save("../dataset/test_labels.npy", y_test)

#  Build 14x14 versions
xtrn = np.zeros((60000,14,14), dtype="float32")
for i in range(60000):
    xtrn[i,:,:] = cv2.resize(x_train[i], (14,14), interpolation=cv2.INTER_LINEAR)
xtst = np.zeros((10000,14,14), dtype="float32")
for i in range(10000):
    xtst[i,:,:] = cv2.resize(x_test[i], (14,14), interpolation=cv2.INTER_LINEAR)

np.save("../dataset/train_images_small.npy", xtrn)
np.save("../dataset/test_images_small.npy", xtst)

