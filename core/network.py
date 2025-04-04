import json
import os

import numpy as np

from core.logger import Logger


def _xavier_uniform(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_out, n_in))


def _xavier_normal(n_in, n_out):
    stddev = np.sqrt(2 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * stddev


def _softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def _leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def _leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


class Network:
    drop_factor = 1.0
    epochs_drop = 1000

    def __init__(self, layer_sizes=[], learning_rate=0.01):
        np.random.seed(42)
        self.__lr = learning_rate
        self.__training_data = np.array([])
        self.__layer_sizes = layer_sizes
        self.__outputs = np.array([])

        self.__weights = [
            _xavier_normal(layer_sizes[i + 1], layer_sizes[i]).T
            for i in range(len(layer_sizes) - 1)
        ]
        self.__biases = [
            np.zeros(layer_sizes[i + 1]).reshape(-1, 1)
            for i in range(len(layer_sizes) - 1)
        ]

    def __forward_prop(self, x):
        activations = [x]
        pre_activations = []

        for w, b in zip(self.__weights, self.__biases):
            z = np.dot(w, activations[-1]) + b
            pre_activations.append(z)
            a = _leaky_relu(z)
            activations.append(a)

        # activations[-1] = np.clip(activations[-1], -1.0, 1.0)
        return pre_activations, activations

    def __back_prop(self, m, pre_activations, activations, y):
        dz = activations[-1] - y
        dw = np.dot(dz, activations[-2].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dws = [dw]
        dbs = [db]

        for i in range(len(self.__weights) - 2, -1, -1):
            dz = np.dot(self.__weights[i + 1].T, dz) * _leaky_relu_derivative(activations[i + 1])
            dw = np.dot(dz, activations[i].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            dws.insert(0, dw)
            dbs.insert(0, db)

        max_grad = 5.0
        dws = [np.clip(dw, -max_grad, max_grad) for dw in dws]
        dbs = [np.clip(db, -max_grad, max_grad) for db in dbs]

        return dws, dbs

    def load_training_data(self, inputs, outputs):
        self.__training_data = np.array(inputs)
        self.__outputs = np.array(outputs)

    def train(self, iterations=1):
        logger = Logger()
        samples = self.__training_data.shape[1]

        logger.epoch_logger(10, iterations)
        for i in range(iterations):
            lr = self.__lr * (self.drop_factor ** (i // self.epochs_drop))
            pre_activations, activations = self.__forward_prop(self.__training_data)
            dws, dbs = self.__back_prop(samples, pre_activations, activations, self.__outputs)

            for j in range(len(self.__weights)):
                self.__weights[j] -= lr * dws[j]
                self.__biases[j] -= lr * dbs[j]
            logger.epoch_log(i)

    def predict(self, inputs):
        _, activations = self.__forward_prop(inputs)
        return np.squeeze(activations[-1])

    def serialize(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        network_data = {
            "layer_sizes": self.__layer_sizes,
            "weights": [w.tolist() for w in self.__weights],
            "biases": [b.tolist() for b in self.__biases]
        }
        json.dump(network_data, open(filename, "w"))

    def deserialize(self, filename):

        if not os.path.isfile(filename):
            print(f"File {filename} does not exist. You should train network first.")
            return

        serialized = json.loads(open(filename, "rb").read())
        self.__layer_sizes = serialized["layer_sizes"]
        self.__weights = [
            np.array(w).reshape(self.__layer_sizes[i + 1], self.__layer_sizes[i])
            for i, w in enumerate(serialized["weights"])
        ]
        self.__biases = [np.array(b) for b in serialized["biases"]]
