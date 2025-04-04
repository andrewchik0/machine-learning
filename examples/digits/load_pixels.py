import csv
import os

import numpy as np


def load_pixels(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist. Please get it from MNIST dataset.")
        return

    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)

        data = np.array([row for row in reader], dtype=float)  # Load everything at once

    digits = data[:, 0].astype(int)
    pixels = data[:, 1:]

    pixels = pixels / 255.0

    return pixels, digits

