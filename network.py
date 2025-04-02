import numpy as np

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


class Network:
    drop_factor = 1.0
    epochs_drop = 1000

    def __init__(self, layer_sizes, learning_rate=0.01):
        np.random.seed(42)
        self.__lr = learning_rate
        self.__training_data = np.array([])
        self.__outputs = np.array([])

        self.weights = [
            _xavier_normal(layer_sizes[i + 1], layer_sizes[i]).T
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [
            np.zeros(layer_sizes[i + 1]).reshape(-1, 1)
            for i in range(len(layer_sizes) - 1)
        ]

    def __forward_prop(self, x):
        activations = [x]
        pre_activations = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            pre_activations.append(z)
            a = _leaky_relu(z)
            activations.append(a)

        activations[-1] = np.clip(activations[-1], 0.0, 1.0)  # Output layer
        return pre_activations, activations

    def __back_prop(self, m, pre_activations, activations, y):
        dz = activations[-1] - y
        dw = np.dot(dz, activations[-2].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dws = [dw]
        dbs = [db]

        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(self.weights[i + 1].T, dz) * activations[i + 1] * (1 - activations[i + 1])
            dw = np.dot(dz, activations[i].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            dws.insert(0, dw)
            dbs.insert(0, db)

        return dws, dbs

    def load_training_data(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.__training_data = np.array(inputs)
        self.__outputs = np.array(outputs)

    def train(self, iterations):
        samples = self.__training_data.shape[1]

        for i in range(iterations):
            lr = self.__lr * (self.drop_factor ** (i // self.epochs_drop))
            pre_activations, activations = self.__forward_prop(self.__training_data)
            dws, dbs = self.__back_prop(samples, pre_activations, activations, self.__outputs)

            for j in range(len(self.weights)):
                self.weights[j] -= lr * dws[j]
                self.biases[j] -= lr * dbs[j]

    def predict(self, inputs):
        _, activations = self.__forward_prop(inputs)
        return np.squeeze(activations[-1])