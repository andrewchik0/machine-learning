from core.network import *


if __name__ == '__main__':
    net = Network([2, 1000, 100, 1])

    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    total_input = np.array([a, b])
    y_xor = np.array([[0, 1, 1, 0]])

    net.load_training_data(total_input, y_xor)
    net.train(iterations=100000)

    net.serialize("trained/binary_xor.json")

