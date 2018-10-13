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


def get_both_database(data1, data2, _size=300):
    channel_1 = data1[:, 1:]
    channel_2 = data2[:, 1:]
    output = np.array([])
    for i in range(6):
        temp1 = channel_1[i*_size:(i * _size) + _size, :]
        temp2 = channel_2[i*_size:(i * _size) + _size, :]
        label = np.ones((2 * _size, 1)) * i
        temp_out = np.concatenate((temp1, temp2), axis=0)
        temp_out = np.concatenate((label, temp_out), axis=1)
        if not output.any():
            output = temp_out
        else:
            output = np.concatenate((output, temp_out), axis=0)
        np.save('Database_2_total', output)

def normalizar(data_file):
    mean = data_file.mean(axis=0)
    std = data_file.std(axis=0)
    return (data_file - mean) / std


def extract_features(mode, ncmp):
    feats = None
    labels = None
    names = None
    if f'{mode}.npy' or f"features_{mode}.npy" in os.listdir("./"):

        # # Para futuro.. por ahora simple nomas
        # load = input("[Seleccion] Se han detectado archivos '.npy'. Deseas cargarlos para ahorrar tiempo? y/n: ")
        if not 'Database' in mode:
            mode = f"features_{mode}"
        data = np.load(mode + '.npy')
        if 'ch1' in mode:
            names = np.load('Database_2_names_ch1.npy')
        if 'ch2' in mode:
            names = np.load('Database_2_names_ch2.npy')

        labels, feats = np.array(data[:, :1]).astype(int),\
                        np.array(data[:, 1:]).astype(int)

        print("[Seleccion] Normalizando...")
        # scaler = StandardScaler().fit(feats)  # Normalizacion
        # feats = scaler.transform(feats)

        do_pca = input('Desea aplicar PCA a los datos? y/n: ')
        if do_pca.lower() == 'y':
            print("[Seleccion] Aplicando PCA...")
            pca = PCA(n_components=ncmp, svd_solver='full')
            pca.fit(feats)
            feats = pca.transform(feats)

            return feats, labels, None, pca, names
##        save = input("Deseas guardar los resultados para futuras instancias? y/n: \n")
##        if save.lower() == 'y':
##            np.save("{}feats_data".format(mode+"_"), feats)
##            pickle.dump(pca, open("{}pca.p".format(mode+"_"), "wb"))
##            pickle.dump(scaler, open("{}scaler.p".format(mode+"_"), "wb"))
    else:
        #Extraer nuevas features.. para futuro tambien
        pass

    return feats, labels, None, None, names

def train_predict(feats_dict, ncmp):
    print("[Clasificacion] Entrenando clasificadores...")

    feats, labels = feats_dict["feats"], feats_dict["labels"]
    labels = np.ravel(labels)

    lda = LDA(n_components=ncmp)
    lda.fit(feats, labels)

    svm = SVC(kernel='linear', decision_function_shape='ovr')
    svm.fit(feats, labels)

    knn = KNN(n_neighbors=5)
    knn.fit(feats, labels)

    return lda, svm, knn

def classify(Test_data, names, lda, svm, knn):

    print("[Clasificacion] Probando clasificadores...")

    lda_predict = lda.predict(Test_data)[0]
    svm_predict = svm.predict(Test_data)[0]
    knn_predict = knn.predict(Test_data)[0]

    print(f"LDA: {names[lda_predict]} || SVM:  {names[svm_predict]} ||"
          f" KNN: {names[knn_predict]}")
    print()

    return lda

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

    data1 = np.load('Database_2_total_ch1.npy')
    data2 = np.load('Database_2_total_ch2.npy')
    get_both_database(data1, data2, _size=300)

    input()

    """Funcion principal del programa"""

    ncmp = 2500

    # Training Feature Extraction
    print("\n[Extraccion Features] Procesando datos...")
    file = input('Ingrese sector de los datos: ')
    feats, labels, scaler, train_pca, names = extract_features(file, ncmp=ncmp)

    feats_dict = {'feats': feats, 'labels': labels}
    trained_pred = train_predict(feats_dict, ncmp=ncmp)
    continuar = True
    while continuar:
        test_line = get_measure(length=2500)
        # option = input('Choose finger / fist: ')
        # test_line = create_measure(option, peak_val=1, duration=300, length=2500)
        if scaler or train_pca:
            print('Aplicando PCA a test line')
            _size = feats_dict["feats"].shape[0]
            test_line = np.tile(test_line, (_size, 1))
            print('Forma Test line: ', test_line.shape)
            test_line = train_pca.transform(test_line)
            test_line = test_line[:1, :]
        test_line = test_line.reshape(1, -1)
        classify(test_line, names, *trained_pred)

        continuar = True if input('Desea continuar? y/n: ') == 'y' else False

