from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import sleep
import numpy as np
import random as rn
import serial
import os


def normalizar(data_file):
    mean = data_file.mean(axis=0)
    std = data_file.std(axis=0)
    return (data_file - mean) / std

def load_data():
    Dat2_data = np.load('Database_2_total.npy')
    Dat1_data_ch1 = np.load('Database_1_total_ch1.npy')
    Dat1_data_ch2 = np.load('Database_1_total_ch2.npy')
    _names = np.load('database_2_names_ch1.npy')

    total_data = np.array([])
    for i in range(6):
        if not total_data.any():
            total_data = np.concatenate((Dat2_data[i * 600:i * 600 + 600, :],
                                         Dat1_data_ch1[i * 150:i * 150 + 150, :2501],
                                         Dat1_data_ch2[i * 150:i * 150 + 150, :2501]),
                                        axis=0)
        else:
            total_data = np.concatenate((total_data,
                                         Dat2_data[i * 600:i * 600 + 600, :],
                                         Dat1_data_ch1[i * 150:i * 150 + 150, :2501],
                                         Dat1_data_ch2[i * 150:i * 150 + 150, :2501]),
                                        axis=0)
    return total_data

def partition_data(feats, moves, test_size=0.2):
    names2num = {'cyl': 0, 'hook': 1, 'tip': 2, 'palm': 3, 'spher': 4, 'lat': 5}
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

def pca_features(feats):
    ncmp = feats.shape[1] - 500

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


    print(f"\nLDA: {result[0]} || SVM:  {result[1]} ||"
          f" KNN: {result[2]}")

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

def create_measure(_type, peak_val, duration, length=5000):
    measure = np.zeros((1, length))
    if _type == 'finger':
        measure[:, 1000: 1000 + duration] = np.ones((1, duration)) * \
                                            peak_val * rn.uniform(0.6, 1.2)
    if _type == 'fist':
        measure[:, 100:] = np.ones((1, length - 100)) * peak_val * \
                           rn.uniform(0.6, 1.2)
    return measure


if __name__ == '__main__':

    moves = ['palm', 'spher']

    train, test = partition_data(load_data(), moves=moves, test_size=0.2)

    train_feats, train_labels = train[:, 1:], train[:, :1]
    test_feats, test_labels = test[:, 1:], test[:, :1]

    ncmp = train_feats.shape[1] - 500

    pca_bool = True if input('Desea aplicar PCA? [y/~]: ') == 'y' else False
    if pca_bool:
        train_feats, pca = pca_features(train_feats)
        test_feats = pca.transform(test_feats)

    lda, svm, knn = train_predict(train_feats, train_labels, ncmp=ncmp)

    result = classify(test_feats, test_labels, lda=lda, svm=svm, knn=knn)

    print('Acc obtenida en LDA: {} SVM: {} KNN: {}'.format(*result))


