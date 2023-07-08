#!/Users/simonwittum/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Description
===========

This file contains an implementation for a neural network with three layers from scratch,
solving the MNIST classification problem. The network makes use of one mid layer with 10 nodes,
using ReLU and softmax as activation functions. The optimization of the weights and biases happens
with the technique of gradient descent, minimizing the value of the cross entropy loss function
which measures the deviation from the current predictions done by the network from the actual
classifications.

@author: Simon Wittum
MIT license
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

# Load training data
data_train = pd.read_csv("/Users/simonwittum/Documents/Programmieren/Python/MachineLearning/NeuralNetwork/data/mnist_train.csv")
data_train = np.array(data_train)
np.random.shuffle(data_train)
data_train = data_train.T
y_train = data_train[0, 1000:]
x_train = data_train[1:, 1000:] / 255.


# Load testing data
data_test = pd.read_csv("./data/mnist_train.csv")
data_test = np.array(data_test)
np.random.shuffle(data_test)
data_test = data_test.T
y_test = data_test[0, :]
x_test = data_test[1:, :] / 255.

W1 = np.loadtxt("./Params/W1.txt").reshape(10, 784)
b1 = np.loadtxt("./Params/b1.txt").reshape(10, 1)
W2 = np.loadtxt("./Params/W2.txt").reshape(10, 10)
b2 = np.loadtxt("./Params/b2.txt").reshape(10, 1)

neural_network = NeuralNetwork()
neural_network.W1 = W1
neural_network.b1 = b1
neural_network.W2 = W2
neural_network.b2 = b2


def main():
    start = time.time()

    _, _, _, A2 = neural_network.forward(x_train)
    # We do not need to randomize the index as data is
    # being shuffled at beginning
    neural_network.evaluate(np.random.randint(len(y_test)), x_test, y_test)
    # print(neural_network)
    end = time.time()
    print("This Script Took {:.2f}s to Compute.".format(end - start))
    plt.show()

    return None


if __name__ == "__main__":
    print("Entering Script...\n")
    main()
    print("Exiting Script - Thank You and Goodbye!")
