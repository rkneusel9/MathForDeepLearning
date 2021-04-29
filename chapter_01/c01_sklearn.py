# c01_sklearn.py

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

d = load_digits()
digits = d["data"]
labels = d["target"]

N = 200
idx = np.argsort(np.random.random(len(labels)))
x_test, y_test = digits[idx[:N]], labels[idx[:N]]
x_train, y_train = digits[idx[N:]], labels[idx[N:]]

clf = MLPClassifier(hidden_layer_sizes=(128,))
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
pred = clf.predict(x_test)
err = np.where(y_test != pred)[0]
print()
print("score      : ", score)
print("errors:")
print("  actual   : ", y_test[err])
print("  predicted: ", pred[err])
print()

