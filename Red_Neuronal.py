#Serial bus and others
from functools import reduce
import serial
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

ser = serial.Serial("COM4", baudrate=115200, timeout=1)


def read_line():
    line = ser.readline()
    data = str(line.decode('cp437')).strip('\r0\r\n')
    if not data.isnumeric():
        data = 0
    return float(data)


def debouncer():
    test = []
    for _ in range(5):
        if read_line():
            test.append(True)
        else:
            test.append(False)
    eval = reduce(lambda x, y: x and y, test)
    return eval


#Cambiar normalizacion o.. normalizar los datos de entrenamiento tambien
def normalizar(data_line):
    mean = data_line.mean(axis=0)
    std = data_line.std(axis=0)
    return (data_line - mean) / std


def normalizar_matriz(matriz):
    y, _ = matriz.shape
    for i in range(y):
        matriz[i, :] = normalizar(matriz[i, :])
    return matriz


def load_train_data():
    total = input('Desea cargar todos los datos o solo un channel? [t/u]: ')
    if total.lower() == 't':
        data = np.load('Database_2_total.npy')
        _names = np.load('database_2_names_ch1.npy')
    else:
        path = input('Ingrese Dataset y channel: ')
        dataset, channel = path.split()
        data = np.load('{}_total_{}.npy'.format(dataset, channel))
        _names = np.load('{}_names_{}.npy'.format(dataset, channel))

    feats, labels = data[:, 1:], data[:, :1]
    feats = normalizar_matriz(feats)
    return feats, labels, _names


def get_measure(length):
    test_feat = []
    print('Tomando medicion !!')
    while len(test_feat) < length:  #esa cantidad mas 1 del label
        data = read_line()
        test_feat.append(float(data))
    return normalizar(np.array(test_feat))


if __name__ == '__main__':

    train_data, train_labels, names = load_train_data()

    with tf.device('/device:GPU:1'):
        model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(6, activation=tf.nn.softmax)])

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_data, train_labels, epochs=15)

        continuar = True
        while continuar:

            #Una fila de datos
            while not debouncer():
                test_data = get_measure(2500)
            # Add the image to a batch where it's the only member.
            test_data = (np.expand_dims(test_data, 0))
            if test_data.any():

                pred_single = model.predict(test_data)

                ind = np.argmax(pred_single)
                print(pred_single)
                print(names[ind] + '\n')


                continuar = True if input('Desea continuar? y/n: ') == 'y' else False


