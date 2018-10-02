from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import sleep
import numpy as np
import serial
import os





def extract_features(mode, ncmp):
    feats = ''
    labels = ''
    if 'features_{}.npy'.format(mode) in os.listdir("./"):

        # # Para futuro.. por ahora simple nomas
        # load = input("[Seleccion] Se han detectado archivos '.npy'. Deseas cargarlos para ahorrar tiempo? y/n: ")

        data = np.load(f"features_{mode}.npy")
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

            return feats, labels, None, pca
##        save = input("Deseas guardar los resultados para futuras instancias? y/n: \n")
##        if save.lower() == 'y':
##            np.save("{}feats_data".format(mode+"_"), feats)
##            pickle.dump(pca, open("{}pca.p".format(mode+"_"), "wb"))
##            pickle.dump(scaler, open("{}scaler.p".format(mode+"_"), "wb"))
    else:
        #Extraer nuevas features.. para futuro tambien
        pass

    return feats, labels, None, None



def classify(feats_dict, Test_data, ncmp):
    print("[Clasificacion] Entrenando clasificadores...")

    feats, labels = feats_dict["feats"], feats_dict["labels"]
    labels = np.ravel(labels)

    lda = LDA(n_components=ncmp)
    lda.fit(feats, labels)

    svm = SVC(kernel='linear', decision_function_shape='ovr')
    svm.fit(feats, labels)

    knn = KNN(n_neighbors=5)
    knn.fit(feats, labels)

    print("[Clasificacion] Probando clasificadores...")

    lda_predict = lda.predict(Test_data)[0]
    svm_predict = svm.predict(Test_data)[0]
    knn_predict = knn.predict(Test_data)[0]

    print(f"LDA: {lda_predict} || SVM:  {svm_predict} || KNN: {knn_predict}")
    print()

    return lda

def get_measure():
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
    while len(feature_line) < 5000:
        line = ser.readline()
        data = str(line.decode('cp437')).strip('\r0\r\n')
        if not data.isnumeric():
            data = 0
        feature_line.append(data)
    return np.array(feature_line).astype(int)
    print()


if __name__ == '__main__':
    """Funcion principal del programa"""
    ncmp = 5000
    # Training Feature Extraction
    print("\n[Extraccion Features] Procesando datos...")
    file = input('Ingrese sector de los datos: ')
    feats, labels, scaler, train_pca = extract_features(file, ncmp=ncmp)

    feats_dict = {'feats': feats, 'labels': labels}

    continuar = True
    while continuar:
        test_line = get_measure()
        if scaler or train_pca:
            print('Aplicando PCA a test line')
            _size = feats_dict["feats"].shape[0]
            test_line = np.tile(test_line, (_size, 1))
            print('Forma Test line: ', test_line.shape)
            test_line = train_pca.transform(test_line)
            test_line = test_line[:1, :]
        test_line = test_line.reshape(1, -1)
        classify(feats_dict, test_line, ncmp=ncmp)

        continuar = True if input('Desea continuar? y/n: ') == 'y' else False

