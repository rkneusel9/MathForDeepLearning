#
#  Simple matrix-vector operations example
#
#  RTK, 11-Apr-2020 (Happy bday, Peter!)
#  Last update:  11-Apr-2020
#
################################################################

import matplotlib.pylab as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

#  Build the dataset
np.random.seed(8675309)
x0 = np.random.random(50)-0.3
y0 = np.random.random(50)+0.3
x1 = np.random.random(50)+0.3
y1 = np.random.random(50)-0.3
print("x0,y0: %0.6f, %0.6f" % (x0.mean(), y0.mean()))
print("x1,y1: %0.6f, %0.6f" % (x1.mean(), y1.mean()))
print()
x = np.zeros((100,2))
x[:50,0] = x0; x[:50,1] = y0
x[50:,0] = x1; x[50:,1] = y1
y = np.array([0]*50+[1]*50)

#  Randomize and make train/test split
idx = np.argsort(np.random.random(100))
x = x[idx]
y = y[idx]
x_train = x[:75]
y_train = y[:75]
x_test = x[75:]
y_test = y[75:]

#  Show the dataset
plt.plot(x0,y0,marker='o',linestyle='none')
plt.plot(x1,y1,marker='s',linestyle='none')
plt.xlabel(r'$x_0$')
plt.ylabel(r'$x_1$')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("c04_nn.png", dpi=300)

#  Train a simple model
clf = MLPClassifier(hidden_layer_sizes=(5,))
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
prob = clf.predict_proba(x_test)
print("Model accuracy on test set: %0.4f" % score)
W0 = clf.coefs_[0].T
b0 = clf.intercepts_[0].reshape((5,1))
W1 = clf.coefs_[1].T
b1 = clf.intercepts_[1]

print("Weights and biases:")
print(W0)
print(b0)
print()
print(W1)
print(b1)
print()

z = x_test[0].reshape((2,1))
print("x_test:", z)
print("W0 @ z + b0", W0 @ z + b0)
print("a0 = relu(W0 @ z + b0)", np.maximum(0,W0@z+b0))
a0 = np.maximum(0,W0@z+b0)
print("a1 = W1@a0 + b1", W1@a0+b1)
a1 = W1@a0+b1
print("sigmoid(a1)", 1.0/(1.0+np.exp(-a1)))
print()
print("prob: ", prob[0][1])
print("y_test: ", y_test[0])
print()

