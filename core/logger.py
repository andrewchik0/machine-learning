
class Logger(object):

    def __init__(self):
        self.__pass_iterations = 0
        self.__max_iterations = 0
        pass

    def log(self, data):
        print(data)

    def epoch_logger(self, pass_iterations, max_iterations):
        self.__pass_iterations = pass_iterations
        self.__max_iterations = max_iterations

    def epoch_log(self, iteration):
        if iteration % self.__pass_iterations == 0:
            print(f"Epoch: {iteration} ({iteration / self.__max_iterations * 100:.2f}%)")
