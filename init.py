import time

from network import *

def main():

    net = Network([2, 100, 10, 1])

    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    total_input = np.array([a, b])
    y_and = np.array([[0, 1, 1, 0]])

    net.load_training_data(total_input, y_and)
    net.train(iterations=10000)

    print(net.predict([[0], [0]]))
    print(net.predict([[0], [1]]))
    print(net.predict([[1], [0]]))
    print(net.predict([[1], [1]]))


if __name__ == '__main__':
    main()