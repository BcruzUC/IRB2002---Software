#Serial bus and others
from sklearn.preprocessing import Normalizer
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
    try:
        return int(data)
    except ValueError:
        return 0


def debouncer(pulses=5):
    test = []
    for _ in range(pulses):
        if read_line() != 0:
            test.append(True)
        else:
            test.append(False)
    eval = reduce(lambda x, y: x and y, test)
    return eval


#Cambiar normalizacion o.. normalizar los datos de entrenamiento tambien
def normalizar(data):
    scaler = Normalizer(norm='max')
    if len(data[:, 0]) == 1:
        # data = data.reshape(1, -1)
        return scaler.transform(data)
    if len(data)[:, 0] > 1:
        return scaler.transform(data)


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
    feats = normalizar(feats)
    return feats, labels, _names

def load_test_data(path):
    data = np.load(path + '.npy')
    labels, feats = data[:, :1], data[:, 1:2501]
    feats = normalizar(feats)
    return labels, feats


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


def get_measure(length):
    test_feat = []
    print('Tomando medicion !!')
    while len(test_feat) < length:  #esa cantidad mas 1 del label
        data = read_line()
        test_feat.append(float(data))
    test_feat = np.array(test_feat).reshape(1, -1)
    return normalizar(test_feat)


if __name__ == '__main__':

    moves = ['palm', 'spher']

    train, test = partition_data(load_data(), moves=moves, test_size=0.2)

    train_feats, train_labels = train[:, 1:], train[:, :1]
    test_feats, test_labels = test[:, 1:], test[:, :1]

    # plot_data = []
    #
    # for neuron_num in range(5):

    with tf.device('/device:GPU:1'):
        model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)])

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_feats, train_labels, epochs=10)

    test_mode = input('Modo de prueba [sensor/dataset]: ')

    if test_mode.lower() == 'sensor':
        continuar = True
        while continuar:
            #Una fila de datos
            test_data = np.zeros((1, 2500))
            while True:
                if debouncer(pulses=10):
                    test_data = get_measure(2500)
                    break
            # Add the image to a batch where it's the only member.
            # test_data = (np.expand_dims(test_data, 0))
            if test_data.any():
                test_data = test_data.reshape(1, -1)

                pred_single = model.predict(test_data)

                ind = np.argmax(pred_single)
                print(pred_single)
                moves = moves[::-1]
                print(moves[ind] + '\n')


                continuar = True if input('Desea continuar? y/n: ') == 'y' else False

    if test_mode.lower() == 'dataset':
        # Tlabels, Tfeats = load_test_data('Database_1_total_ch2')
        # print(Tfeats.shape)
        test_loss, test_acc = model.evaluate(test_feats, test_labels)

        print('Test accuracy:', test_acc)

        # plot_data.append((neuron_num, test_acc))

    # plt_x = [x[0] for x in plot_data]
    # plt_y = [x[1] for x in plot_data]
    #
    # plt.plot(plt_x, plt_y)
    # plt.show()


