from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import random as rn
import serial
import os


def normalizar(data):
    scaler = Normalizer(norm='l2')
    if len(data[:, 0]) == 1:
        data = data.reshape(1, -1)
        return scaler.transform(data)
    if len(data[:, 0]) > 1:
        return scaler.transform(data)


def load_data(mode='raw'):
    Data1 = np.load('raw_Database_1_total.npy')
    Data2 = np.load('raw_Database_2_total.npy')
    no_move = np.tile(np.load('no_move_100.npy'), (9, 1))

    print(f'data1: {Data1.shape} data2: {Data2.shape}')

    total_data = np.array([])
    for i in range(6):
        if not total_data.any():
            total_data = np.concatenate((Data1[i*300:i*300 + 300, :2501],
                                         Data2[i*600:i*600 + 600, :]), axis=0)
        else:
            total_data = np.concatenate((total_data,
                                         Data1[i*300:i*300 + 300, :2501],
                                         Data2[i*600:i*600 + 600, :]), axis=0)
    if mode == 'raw':
        return np.concatenate((total_data, no_move), axis=0)
    return abs(np.concatenate((total_data, no_move), axis=0))


def partition_data(feats, moves, test_size=0.2):
    names2num = {'cyl': 0, 'hook': 1, 'tip': 2, 'palm': 3, 'spher': 4, 'lat': 5,
                 'no_move': 6}
    mod_feats = np.array([])
    # Seleccionar solo los movimientos indicados en 'moves'
    for name in moves:
        num = names2num[name]
        temp = feats[num * 900:num * 900 + 900, :]
        if not mod_feats.any():
            mod_feats = temp
        else:
            mod_feats = np.concatenate((mod_feats, temp), axis=0)
    # Llenar las listas segun el porcentaje de testeo que queremos
    train = []
    test = []
    for i in range(len(moves)):
        for data in range(900):
            if data < (900 - test_size * 900):
                train.append(feats[data + i * 900, :])
            else:
                test.append(feats[data + i * 900, :])

    train = np.array(train)
    test = np.array(test)

    print("\n[Hold-Out] Paritioning features into 2 arrays with shapes {} and {}\n".format(train.shape, test.shape))

    return train, test


def pca_features(feats, ncmp):
    print("[Seleccion] Aplicando PCA...")
    pca = PCA(n_components=ncmp, svd_solver='full')
    pca.fit(feats)
    feats = pca.transform(feats)
##        save = input("Deseas guardar los resultados para futuras instancias? y/n: \n")
##        if save.lower() == 'y':
##            np.save("{}feats_data".format(mode+"_"), feats)
##            pickle.dump(pca, open("{}pca.p".format(mode+"_"), "wb"))
##            pickle.dump(scaler, open("{}scaler.p".format(mode+"_"), "wb"))
    return feats, pca


def train_predict(feats, labels, ncmp):
    print("[Clasificacion] Entrenando clasificadores...")

    lda = LDA(n_components=ncmp)
    lda.fit(feats, labels)

    svm = SVC(kernel='linear', decision_function_shape='ovr')
    svm.fit(feats, labels)

    knn = KNN(n_neighbors=5)
    knn.fit(feats, labels)

    return lda, svm, knn


def classify(test_data, test_label, lda, svm, knn):
    print("\n[Clasificacion] Probando clasificadores...")
    N, _ = test_data.shape
    result = [0, 0, 0]
    for i, element in enumerate(test_data):
        lda_predict = 1 if lda.predict([element])[0] == test_label[i] else 0
        svm_predict = 1 if svm.predict([element])[0] == test_label[i] else 0
        knn_predict = 1 if knn.predict([element])[0] == test_label[i] else 0

        result[0] += lda_predict
        result[1] += svm_predict
        result[2] += knn_predict

    # print(f"\nLDA: {result[0]} || SVM:  {result[1]} ||"
    #       f" KNN: {result[2]}")

    return np.array(result) / N

def get_measure(length):
    ser = serial.Serial("COM4", baudrate=115200, timeout=1)
    print('Tomando medida en 3..')
    sleep(0.5)
    print('Tomando medida en 2..')
    sleep(0.5)
    print('Tomando medida en 1..')
    sleep(0.5)
    print('¡ Ya !')
    sleep(0.5)
    feature_line = []
    while len(feature_line) < length:
        line = ser.readline()
        data = str(line.decode('cp437')).strip('\r0\r\n')
        if not data.isnumeric():
            data = 0
        feature_line.append(data)
    final_data = np.array(feature_line).astype(int)
    return normalizar(final_data)
    print()


if __name__ == '__main__':

    moves = ['palm', 'spher', 'no_move']

    data = load_data(mode='raw')  # load_data()

    train, test = partition_data(data, moves=moves, test_size=0.2)

    ncmp = [num for num in range(200, 2000, 100)]

    pca_bool = True if input('Desea aplicar PCA? [y/~]: ') == 'y' else False
    acc_stack = []

    for pca_num in ncmp:

        train_feats, train_labels = train[:, 1:], train[:, :1]
        test_feats, test_labels = test[:, 1:], test[:, :1]

        if pca_bool:
            train_feats, pca = pca_features(train_feats, ncmp=pca_num)
            test_feats = pca.transform(test_feats)

        lda, svm, knn = train_predict(train_feats, train_labels, ncmp=pca_num)

        result = classify(test_feats, test_labels, lda=lda, svm=svm, knn=knn)
        result = np.insert(result, 0, pca_num)
        acc_stack.append(result)

        print('Acc obtenida en {} para LDA: {} SVM: {} KNN: {}\n'.format(*result))

    x = np.array([line[0] for line in acc_stack])

    y_lda = np.array([line[1] for line in acc_stack])
    y_svm = np.array([line[2] for line in acc_stack])
    y_knn = np.array([line[3] for line in acc_stack])

    plt.plot(x, y_lda, color='blue', label='LDA')
    plt.plot(x, y_svm, color='green', label='SVM')
    plt.plot(x, y_knn, color='red', label='KNN')

    plt.legend(loc='best')
    plt.show()
