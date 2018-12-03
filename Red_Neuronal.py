#Serial bus and others
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from functools import reduce
import random as rn
import os
import time
import serial

# Keras para red neuronal
import keras as kr

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    scaler = Normalizer(norm='l2')
    if len(data[:, 0]) == 1:
        data = data.reshape(1, -1)
        return scaler.transform(data)
    if len(data[:, 0]) > 1:
        return scaler.transform(data)

def ruido(matrix, seed=0):
    feats, labels = matrix[:, 1:], matrix[:, :1]
    out_matrix = np.array([])
    for row in range(matrix.shape[0]):  #Esta recorriendo las columnas, no las filas.
        rn.seed(seed)
        temp = rn.normalvariate(1, 0.1)
        out_row = feats[row, :].reshape(1, -1) * temp
        if not out_matrix.any():
            out_matrix = out_row
        else:
            out_matrix = np.concatenate((out_matrix, out_row), axis=0)
    return np.concatenate((labels, out_matrix), axis=1)


def load_data_sensor():
    Data1 = np.load('palm_400.npy')
    Data2 = np.load('spher_400.npy')
    Data1_n = ruido(Data1, seed=0)
    Data2_n = ruido(Data2, seed=0)
    Data1_nn = ruido(Data1, seed=42)
    Data2_nn = ruido(Data2, seed=42)

    no_move = np.tile(np.load('no_move_100.npy'), (12, 1))

    all_data = np.concatenate((Data1, Data1_n, Data1_nn, Data2, Data2_n,
                               Data2_nn, no_move), axis=0)

    return all_data

def load_data(mode='raw', length=2500):
    Data1 = np.load('raw_Database_1_total.npy')
    Data2 = np.load('raw_Database_2_total.npy')
    no_move = np.tile(np.load('no_move_300.npy'), (3, 1))

    print(no_move.shape)

    total_data = np.array([])
    for i in range(6):
        if not total_data.any():
            total_data = np.concatenate((Data1[i*300:i*300 + 300, :length+1],
                                         Data2[i*600:i*600 + 600, :length+1]), axis=0)
        else:
            total_data = np.concatenate((total_data,
                                         Data1[i*300:i*300 + 300, :length+1],
                                         Data2[i*600:i*600 + 600, :length+1]), axis=0)
    if mode == 'raw':
        return np.concatenate((total_data, no_move), axis=0)
    return abs(np.concatenate((total_data, no_move), axis=0))


def partition_data(feats, moves, test_size=0.2, mode='ho', length=2500):
    if moves:
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
        feats = mod_feats
    if mode == 'ho':
        train, test = train_test_split(feats, test_size=test_size, random_state=42, shuffle=True)
    else:
        train, test = train_test_split(feats, test_size=test_size, shuffle=True)

    train = np.array(train)[:, :length + 1]
    test = np.array(test)[:, :length + 1]

    print("\n[DATA] Paritioning features into 2 arrays with shapes {} and {}\n".format(train.shape, test.shape))

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
    elif mode == 'abs':
        return normalizar(abs(test_feat))

if __name__ == '__main__':

    moves = ['palm', 'spher', 'no_move']
    save_path = 'model'
    data_length = 1200
    move_debouncer = 2

    data1 = load_data(mode='raw', length=1200)
    # data2 = load_data_sensor()
    train1, test1 = partition_data(data1, moves=moves, test_size=0.2, mode='ho', length=data_length)

    # train2, test2 = partition_data(data2, moves=None, test_size=0.2, mode='ho')

    train = train1 #np.concatenate((train1, train2), axis=0)
    test = test1 #np.concatenate((test1, test2), axis=0)

    train_feats, train_labels = train[:, 1:], train[:, :1]
    test_feats, test_labels = test[:, 1:], test[:, :1]

    train_feats = normalizar(train_feats)
    test_feats = normalizar(test_feats)

    if 'model.json' in os.listdir('./model') and \
            input('[MODELO] Modelo encontrado, desea cargarlo ?[y/n]: ') == 'y':

        json_file = open(f'{save_path}/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = kr.models.model_from_json(loaded_model_json)
        model.load_weights(f"{save_path}/model_weights.h5")
        print('\n[MODELO] Modelo cargado con exito')
        model.compile(optimizer=kr.optimizers.Adam(lr=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    else:

    # plot_data = []
    # for num in (128, 256, 512, 767, 1024):
        model = kr.Sequential()
        model.add(kr.layers.Dense(512, activation='relu'))
        model.add(kr.layers.Dense(3, activation='softmax'))

        model.compile(optimizer=kr.optimizers.Adam(lr=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_feats, train_labels, epochs=10)
        if input('[MODELO] Desea guardar el modelo? [y/n]: ') == 'y':
            print('\n[MODELO] Guardando modelo')
            model_json = model.to_json()
            with open(f"{save_path}/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(f"{save_path}/model_weights.h5")
            print('\n[MODELO] Modelo guardado con exito')


    test_mode = input('Modo de prueba [sensor/dataset]: ')

    if test_mode.lower() == 'sensor':
        continuar = True
        temp_move = []
        while continuar:
            moves = ['palm', 'spher', 'no_move']

            #Una fila de datos
            test_data = np.zeros((1, data_length))
            # while True:
            #     if debouncer(pulses=1):

            while len(temp_move) < move_debouncer:
                test_data = get_measure(data_length)
                test_data = test_data.reshape(1, -1)
                pred_single = model.predict(test_data)

                ind = int(np.argmax(pred_single))
                def_move = moves[ind]
                temp_move.append(def_move)

            test_data = get_measure(data_length)
            test_data = test_data.reshape(1, -1)
            pred_single = model.predict(test_data)

            ind = int(np.argmax(pred_single))
            def_move = moves[ind]
            temp_move.append(def_move)

            temp_move.pop(0)
            temp_move.append(def_move)

            final_move = temp_move[0] if temp_move[0] == temp_move[1] else None

            if final_move:
                print('MOVIENDO: {}'.format(final_move), end='  ')

                if final_move == 'spher':
                    bit_str = str.encode('d110;')
                    # ser.write(bit_str)
                    print('d110;')

                if final_move == 'palm':
                    bit_str = str.encode('d130;')
                    print('d130;')
                    # ser.write(bit_str)

                if final_move == 'no_move':
                    bit_str = str.encode('s')
                    # ser.write(bit_str)
                    print('s;')




                # continuar = True if input('Desea continuar? y/n: ') == 'y' else False

    if test_mode.lower() == 'dataset':
        # Tlabels, Tfeats = load_test_data('Database_1_total_ch2')
        # print(Tfeats.shape)
        test_loss, test_acc = model.evaluate(test_feats, test_labels)

        print('Test accuracy:', test_acc)

        # plot_data.append((num, test_acc))

    # plt_x = [x[0] for x in plot_data]
    # plt_y = [x[1] for x in plot_data]
    #
    # plt.plot(plt_x, plt_y, color='lightblue', label='Acc vs N° neurons')
    # plt.legend(loc='best')
    # plt.show()


