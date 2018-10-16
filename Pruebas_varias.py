import numpy as np
import random as rn
from functools import reduce
from sklearn.preprocessing import Normalizer


def normalizar(data_line):
    scaler = Normalizer(norm='l2')
    return scaler.transform([data_line])


def normalizar_matriz(matriz):
    y, _ = matriz.shape
    for i in range(y):
        matriz[i, :] = normalizar(matriz[i, :])
    return matriz

def norm_matrix(matrix):
    scaler = Normalizer(norm='l2')
    return scaler.transform(matrix)

def read_line():
    return rn.randint(0, 4)

def debouncer(pulses=5):
    test = []
    for _ in range(pulses):
        if read_line():
            test.append(True)
        else:
            test.append(False)
    eval = reduce(lambda x, y: x and y, test)
    return eval


if __name__ == '__main__':

    # test = np.load('Database_2_total.npy')
    # print(test.shape)
    # print([line for line in test if line[0] == 2])
    # print(len([line for line in test if line[0] == 2]))

    print(debouncer())

    norm_test = np.array([[1, 2, 0.5, 2], [2, 1, 0, 2]])
    norm_test = norm_matrix(norm_test)
    print(norm_test)

    line_test = np.array([1, 2, 2, 0])
    print(normalizar(line_test))

