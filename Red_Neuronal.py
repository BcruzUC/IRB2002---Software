#Serial bus and others
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from functools import reduce
import pickle
import os
import time
import serial
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

ser = serial.Serial("COM5", baudrate=115200, timeout=1)

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
    scaler = Normalizer(norm='l2')
    if len(data[:, 0]) == 1:
        data = data.reshape(1, -1)
        return scaler.transform(data)
    if len(data[:, 0]) > 1:
        return scaler.transform(data)


def load_data_sensor():
    Data1 = np.tile(np.load('palm_400.npy'), (3, 1))
    Data2 = np.tile(np.load('spher_400.npy'), (3, 1))
    no_move = np.tile(np.load('no_move_100.npy'), (12, 1))

    all_data = np.concatenate((Data1, Data2, no_move), axis=0)

    return all_data[:, 1:], all_data[:, :1]

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


def partition_data(feats, moves, test_size=0.2, mode='ho'):
    names2num = {'cyl': 0, 'hook': 1, 'tip': 2, 'palm': 3, 'spher': 4, 'lat': 5,
                 'no_move': 6}
    mod_feats = np.array([])
    # Seleccionar solo los movimientos indicados en 'moves'
    for i, name in enumerate(moves):
        num = names2num[name]
        label = np.ones((900, 1)) * i
        temp = feats[num * 900:num * 900 + 900, :]
        temp[:, 0] = label[:, 0]
        if not mod_feats.any():
            mod_feats = temp
        else:
            mod_feats = np.concatenate((mod_feats, temp), axis=0)

    if mode == 'ho':
        train, test = train_test_split(mod_feats, test_size=test_size, random_state=42, shuffle=True)
    else:
        train, test = train_test_split(mod_feats, test_size=test_size, shuffle=True)
    # Llenar las listas segun el porcentaje de testeo que queremos

    # print(test)

    train = np.array(train)
    test = np.array(test)

    print("\n[Hold-Out] Paritioning features into 2 arrays with shapes {} and {}\n".format(train.shape, test.shape))

    return train, test


def get_measure(length, mode='raw'):
    test_feat = []
    print('Tomando medicion !!')
    while len(test_feat) < length:  #esa cantidad mas 1 del label
        data = read_line()
        test_feat.append(float(data))
    test_feat = np.array(test_feat).reshape(1, -1)
    if mode == 'raw':
        return normalizar(test_feat)


if __name__ == '__main__':

    moves = ['palm', 'spher', 'no_move']
    #
    # data = load_data(mode='raw')   # load_data()
    #
    # train, test = partition_data(data, moves=moves, test_size=0.2, mode='random')
    #
    # train_feats, train_labels = train[:, 1:], train[:, :1]
    # test_feats, test_labels = test[:, 1:], test[:, :1]
    #
    # train_feats = normalizar(train_feats)
    # test_feats = normalizar(test_feats)

    train_feats, train_labels = load_data_sensor()

    # plot_data = []
    #
    # for neuron_num in range(5):

    if 'NN_model_sensor.p' in os.listdir('./'):
        if input('Modelo encontrado, desea cargarlo ?[y/n]: ') == 'y':
            model = pickle.load(open('NN_model_sensor.p', 'rb'))
    else:
        with tf.device('/device:GPU:1'):
            model = keras.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(len(moves), activation=tf.nn.softmax)])

            model.compile(optimizer=tf.train.AdamOptimizer(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_feats, train_labels, epochs=10)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, 'NN_modelo_sensor')


    test_mode = 'sensor'    # input('Modo de prueba [sensor/dataset]: ')

    if test_mode.lower() == 'sensor':
        continuar = True
        while continuar:

            #Una fila de datos
            test_data = np.zeros((1, 2500))
            while True:
                if debouncer(pulses=1):
                    test_data = get_measure(2500)
                    break
            # Add the image to a batch where it's the only member.
            # test_data = (np.expand_dims(test_data, 0))
            if test_data.any():
                test_data = test_data.reshape(1, -1)

                pred_single = model.predict(test_data)

                ind = int(np.argmax(pred_single))
                print(pred_single)
                def_move = moves[ind]
                print('MOVIENDO: ', def_move)

                if def_move == 'spher':
                    bit_str = str.encode('d')
                    ser.write(bit_str)
                if def_move == 'palm':
                    bit_str = str.encode('d')
                    ser.write(bit_str)

                if def_move == 'no_move':
                    bit_str = str.encode('s')
                    ser.write(bit_str)




                # continuar = True if input('Desea continuar? y/n: ') == 'y' else False

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


