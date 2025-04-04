from core.network import *

from load_pixels import *

def digits_to_nodes(digits_list):
    result_nodes = np.zeros((len(digits_list), 10))

    for index, digit in enumerate(digits_list):
        result_nodes[index][digit] = 1

    return result_nodes.T

if __name__ == '__main__':
    net = Network([784, 128, 64, 10], learning_rate=0.0001)

    pixels, digits = load_pixels("train.csv")

    # Network training on single digit
    test_data = np.array([pixels[1]]).T
    test_output = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    net.load_training_data(test_data, test_output)

    # Network training on full data
    data = pixels.T
    output = digits_to_nodes(digits)
    net.load_training_data(data, output)
    net.train(iterations=10000)

    net.serialize("../trained/handwritten_digits.json")

