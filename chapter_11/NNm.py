#
#  file:  NNm.py  (w/momentum)
#
#  Generic fully connected neural network code using NumPy.
#
#  Based on code by Omar Aflak,
#
#  https://github.com/OmarAflak/Medium-Python-Neural-Network
#
#  used and modified with his permission.
#
#  RTK, 03-Feb-2021
#  Last update:  18-Feb-2021
#
################################################################

import numpy as np

#  Activation function and derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

#  Loss function and derivative
def mse(y_true, y_pred):
    return (0.5*(y_true - y_pred)**2).mean()

def mse_prime(y_true, y_pred):
    return y_pred - y_true


################################################################
#  ActivationLayer
#
class ActivationLayer:
    def forward(self, input_data):
        self.input = input_data
        return sigmoid(input_data)

    def backward(self, output_error):
        return sigmoid_prime(self.input) * output_error
    
    def step(self, eta):
        return


################################################################
#  FullyConnectedLayer
#
class FullyConnectedLayer:
    def __init__(self, input_size, output_size, momentum=0.0):
        #  for accumulating error over a minibatch
        self.delta_w = np.zeros((input_size, output_size))
        self.delta_b = np.zeros((1,output_size))
        self.passes = 0

        #  initialize the weights and biases w/small random values
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
        #  initial velocities
        self.vw = np.zeros((input_size, output_size))
        self.vb = np.zeros((1, output_size))
        self.momentum = momentum

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        #  accumulate the error over the minibatch
        self.delta_w += np.dot(self.input.T, output_error)
        self.delta_b += output_error
        self.passes += 1
        return input_error

    def step(self, eta):
        #  update the weights and biases by the mean error
        #  over the minibatch
        self.vw = self.momentum * self.vw - eta * self.delta_w / self.passes
        self.vb = self.momentum * self.vb - eta * self.delta_b / self.passes
        self.weights = self.weights + self.vw
        self.bias = self.bias + self.vb

        #  reset for the next minibatch
        self.delta_w = np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)
        self.passes = 0


################################################################
#  Network
#
class Network:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = []
        for i in range(input_data.shape[0]):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, minibatches, learning_rate, batch_size=64):
        for i in range(minibatches):
            err = 0

            # select a random minibatch
            idx = np.argsort(np.random.random(x_train.shape[0]))[:batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            for j in range(batch_size):
                # forward propagation
                output = x_batch[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # accumulate loss
                err += mse(y_batch[j], output)

                # backward propagation
                error = mse_prime(y_batch[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
            
            #  update weights and biases
            for layer in self.layers:
                layer.step(learning_rate)

            # report mean loss over minibatch
            if (self.verbose) and ((i%10) == 0):
                err /= batch_size
                print('minibatch %5d/%d   error=%0.9f' % (i, minibatches, err))

# end NNm.py

