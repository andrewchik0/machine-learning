from core.network import *


if __name__ == '__main__':
    net = Network()
    net.deserialize("trained/binary_xor.json")

    print(f"0 XOR 0: {net.predict([[0], [0]]):.14f}")
    print(f"0 XOR 1: {net.predict([[0], [1]]):.14f}")
    print(f"1 XOR 0: {net.predict([[1], [0]]):.14f}")
    print(f"1 XOR 1: {net.predict([[1], [1]]):.14f}")

