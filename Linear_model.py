from sklearn import linear_model as lm
from sklearn.preprocessing import Normalizer
import numpy as np
import random as rn
import matplotlib.pyplot as plt



def normalizar(data):
    scaler = Normalizer(norm='max')
    if len(data) == 1:
        return scaler.transform([data])
    if len(data) > 1:
        return scaler.transform(data)


def load_data(mode='raw'):
    Data1 = np.load('raw_Database_1_total.npy')
    Data2 = np.load('raw_Database_2_total.npy')
    no_move = np.tile(np.load('no_move_100.npy'), (9, 1))

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
    # feats = normalizar(feats)
    return feats, labels, _names

def load_test_data(path):
    data = np.load(path + '.npy')
    labels, feats = data[:, :1], data[:, 1:2501]
    # feats = normalizar(feats)
    return feats, labels


def train_classifiers(feats, zero_num=300, alpha=0.01):

    N, M = feats.shape

    y = np.ones((N//2, 1))                                   #Evaluar valores en los test
    y[N//2 - zero_num:, :] = np.zeros((zero_num, 1))
    spher_class = lm.Ridge(alpha=alpha, normalize=True)
    palm_class = lm.Ridge(alpha=alpha, normalize=True)
    no_move_class = lm.Ridge(alpha=alpha, normalize=True)

    spher_class.fit(feats[:N//2, :], y)  #clase 5
    palm_class.fit(feats[N//2:N, :], y)   #clase 4
    no_move_class.fit(feats[N//2:N, :], y)         #clase 1

    return spher_class, palm_class, no_move_class

def classify(test_data, test_label, spher, palm, no_move):
    acc_list = [0, 0, 0]
    fails = [0, 0, 0]
    for i, line in enumerate(test_data):
        # spher_pred = spher.predict(line)
        # palm_pred = palm.predict(line)
        # cyl_pred = cyl.predict(line)
        line = line.reshape(1, -1)
        predictions = [spher.predict(line),
                       palm.predict(line),
                       no_move.predict(line)]
        # print(predictions)

        pred_ind = predictions.index(max(predictions))

        if pred_ind == test_label[i]:
            acc_list[pred_ind] += 1
        else:
            fails[pred_ind] += 1
    # print(fails)

    return np.array(acc_list) / (test_data.shape[0] / len(acc_list))



if __name__ == '__main__':
    moves = ['palm', 'spher', 'no_move']

    feats = load_data(mode='abs')

    train, test = partition_data(feats, moves=moves,
                                           test_size=0.3)
    train_feat, train_label = train[:, 1:], train[:, :1]
    test_feat, test_label = test[:, 1:], test[:, :1]

    N, M = train_feat.shape
    max_by_num = []
    n_list = list(range(200, N//2 - 100, 10))
    for n in n_list:

        print(f'\nRealizando iteracion: {n}')

        spher, palm, no_move = train_classifiers(train_feat, zero_num=n, alpha=0.001)

        acc_list = classify(test_feat, test_label, spher=spher, palm=palm, no_move=no_move)

        max_by_num.append(acc_list)

        # for i in range(3):
        #     print(f'Acc en {names[i]}: {acc_list[i]}')

    avg_by_num = [sum(acc)/3 for acc in max_by_num]
    ind_max_by_num = avg_by_num.index(max(avg_by_num))
    print(f'El maximo valor encontrado en: {n_list[ind_max_by_num]} con '
          f'{max_by_num[ind_max_by_num]}')



    tot_x = np.array(n_list)

    avg_y = np.array(avg_by_num)
    spher_y = [x[0] for x in max_by_num]
    palm_y = [x[1] for x in max_by_num]
    no_move_y = [x[2] for x in max_by_num]

    plt.plot(tot_x, avg_y, color='blue', label='Promedio esférica - palmar')
    plt.legend(loc='best')
    plt.show()
    plt.plot(tot_x, spher_y, color='green', label='Precisión esférica')
    plt.plot(tot_x, palm_y, color='red', label='Precisión palmar')
    plt.plot(tot_x, no_move_y, color='blue', label='Precisión no-movimiento')
    plt.legend(loc='best')
    plt.show()



