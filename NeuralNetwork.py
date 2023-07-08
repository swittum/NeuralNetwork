#!/Users/simonwittum/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
# author: Simon Wittum

import sys
import time
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load training data
data_train = pd.read_csv("./data/mnist_train.csv")
data_train = np.array(data_train)
np.random.shuffle(data_train)
data_train = data_train.T
y_train = data_train[0, :]
x_train = data_train[1:, :] / 255.

n, m = x_train.shape


# Load testing data
data_test = pd.read_csv("./data/mnist_train.csv")
data_test = np.array(data_test)
np.random.shuffle(data_test)
data_test = data_test.T
y_test = data_test[0, :]
x_test = data_test[1:, :] / 255.


def ReLU(Z: np.ndarray) -> np.ndarray:
    """ 
    Implementation of ReLU activation function
    """
    return np.maximum(Z, 0)


def d_ReLU(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU 
    """
    return Z > 0


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Implementation of softmax activation function
    """
    return np.exp(Z) / sum(np.exp(Z))


class NeuralNetwork:

    def __init__(self):
        """
        Constructor of 'NeuralNetwork' class
        """
        print('Initializing neural network')
        self.init_network()

    def __repr__(self):
        """
        String representation of 'NeuralNetwork' class
        """
        out = 'Network parameters:\n#layers: 2\n#nodes layer1: 10\n#nodes layer2: 10\n'
        return out

    def init_network(self):
        """
        Initialize weights and biases for network with two hidden layers
        """
        self.W1 = np.random.random((10, n)) - .5
        self.b1 = np.random.random((10, 1)) - .5
        self.W2 = np.random.random((10, 10)) - .5
        self.b2 = np.random.random((10, 1)) - .5

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward propagation: Calculate output of network for given weights
        and biases
        """
        Z1 = np.matmul(self.W1, X) + self.b1
        A1 = ReLU(Z1)
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def one_hot_encode(self, Y: np.ndarray) -> np.ndarray:
        """
        One hot encode output values
        """
        n = np.max(Y)
        m = len(Y)
        ohe_Y = np.zeros((n+1, m))
        ohe_Y[Y, np.arange(m)] = 1
        return ohe_Y

    def backward(self, Z1: np.ndarray, A1: np.ndarray, A2: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward propagation to trace back change in output layer
        to deviations in of weights in hidden layers
        """
        # n, m = X.shape
        # Y = self.one_hot_encode(Y)
        # dZ2 = 1 * (A2 - Y)    # used to be A2-Y
        # dW2 = 1/m * np.matmul(dZ2, A1.T)
        # db2 = 1/m * np.sum(dZ2)
        # dZ1 = np.matmul(self.W2.T, dZ2) * d_ReLU(Z1)
        # dW1 = 1/m * np.matmul(dZ1, X.T)
        # db1 = 1/m * np.sum(dZ1)

        n, m = X.shape
        Y = self.one_hot_encode(Y)
        dZ2 = 1/m * (A2 - Y)    # used to be A2-Y
        dW2 = np.matmul(dZ2, A1.T)
        db2 = np.sum(dZ2)
        dZ1 = np.matmul(self.W2.T, dZ2) * d_ReLU(Z1)
        dW1 = np.matmul(dZ1, X.T)
        db1 = np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, alpha: float) -> None:
        """
        Update weights and biases using deviations of weights
        calculated with 'backward' method
        """
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def predictions(self, A2: np.ndarray) -> np.ndarray:
        """
        Calculate predictions of network
        """
        _, m = A2.shape
        predictions = np.zeros(m)
        for i in range(m):
            predictions[i] = np.argmax(A2[:, i])
        return predictions

    def accuracy(self, predictions: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Get current accuracy of network - this should be looked
        at for both testing and training data to prevend over- 
        and underfitting
        """
        n = predictions.size
        counts = 0
        for i in range(n):
            if predictions[i] == Y[i]:
                counts += 1

        acc = 100 * counts / n
        return acc

    def accuracy_testing(self, X: np.ndarray) -> np.ndarray:
        _, _, _, A2 = self.forward(X)
        prediction_ar = self.predictions(A2)
        acc = self.accuracy(prediction_ar, y_test)
        return acc

    def train(self, X: np.ndarray, Y: np.ndarray, alpha: float, cycles: int) -> None:

        reset = round(.05 * cycles)

        n, m = X.shape
        accuracy_ar = np.zeros(cycles)
        accuracy_ar_test = np.zeros(cycles)

        self.init_network()

        counts = 0

        for i in range(cycles):
            Z1, A1, Z2, A2 = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(Z1, A1, A2, X, Y)
            self.update(dW1, db1, dW2, db2, alpha)
            predictions_ar = self.predictions(A2)
            accuracy_el = self.accuracy(predictions_ar, Y)
            accuracy_ar[i] = accuracy_el
            accuracy_el_test = self.accuracy_testing(x_test)
            accuracy_ar_test[i] = accuracy_el_test

            if i % reset == 0:
                if counts > 0:
                    sys.stdout.write("\033[3A\033[2K\r")
                sys.stdout.write("[{}{}]{:.2f}%\n".format(
                    counts*"*", (reset-counts)*"-", (i)/cycles * 100))
                sys.stdout.write("Accuracy on Training Data: {:.2f}%\n".format(
                    accuracy_el))
                sys.stdout.write("Accuracy on Testing Data: {:.2f}%\n".format(
                    accuracy_el_test))

                counts += 1

        sys.stdout.write("[{}{}]{:.2f}%\n".format(
            counts*"*", (reset-counts)*"-", (i+1)/cycles * 100))
        sys.stdout.write("Accuracy on Training Data: {:.2f}%\n".format(
            accuracy_el))
        sys.stdout.write("Accuracy on Testing Data: {:.2f}%\n\n".format(
            accuracy_el_test))

        return accuracy_ar, accuracy_ar_test

    def evaluate(self, k: int, X: np.ndarray, Y: np.ndarray) -> float:
        _, _, _, A2 = self.forward(X)
        predictions_ar = self.predictions(A2)

        fig = plt.imshow(X[:, k].reshape(28, 28))
        plt.axis("off")
        plt.title("The Real Value is {}\nThe Prediction is {}".format(
            Y[k], round(predictions_ar[k])))


def main(save=True) -> int:
    alpha = 0.1
    cycles = 300
    neural_network = NeuralNetwork()

    start = time.time()
    accuracy_ar, accuracy_ar_test = neural_network.train(
        X=x_train, Y=y_train, alpha=alpha, cycles=cycles)
    end = time.time()

    msg = "* This Run Took {:.2f}s to complete *".format(end - start)
    msg_size = len(msg)
    print("*" * msg_size)
    print(msg)
    print("*" * msg_size, end="\n\n")

    iterations = np.arange(len(accuracy_ar))

    fig = plt.figure()
    plt.plot(iterations, accuracy_ar,
             label="Accuracy on Training Data", linewidth=2)
    plt.plot(iterations, accuracy_ar_test,
             label="Accuracy on Testing Data", linewidth=2, linestyle="--")
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("#Cycles / 1")
    plt.ylabel("Accuracy / %")
    plt.title("Convergence of Accuracy")
    plt.show()

    if save:
        np.savetxt("./Params/W1.txt", neural_network.W1)
        np.savetxt("./Params/b1.txt", neural_network.b1)
        np.savetxt("./Params/W2.txt", neural_network.W2)
        np.savetxt("./Params/b2.txt", neural_network.b2)

    # evaluate(0, W1, b1, W2, b2, x_train, y_train)
    # evaluate(1, W1, b1, W2, b2, x_train, y_train)
    # evaluate(2, W1, b1, W2, b2, x_train, y_train)
    # evaluate(3, W1, b1, W2, b2, x_train, y_train)

    plt.show(block=True)

    return 0


if __name__ == "__main__":
    print("Entering Script...\n")
    main()
    print("Exiting Script...")
