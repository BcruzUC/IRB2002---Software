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
    y = np.ones((600, 1))    #Evaluar valores en los test
    y[600 - zero_num:, :] = np.zeros((zero_num, 1))
    spher_class = lm.Ridge(alpha=alpha, normalize=True)
    palm_class = lm.Ridge(alpha=alpha, normalize=True)
    # cyl_class = lm.Ridge(alpha=0.001, normalize=True)

    spher_class.fit(feats[3000:3600, :], y)  #clase 5
    palm_class.fit(feats[2400:3000, :], y)   #clase 4
    # cyl_class.fit(feats[:600, :], y)         #clase 1

    return spher_class, palm_class # cyl_class

def classify(test_data, test_label, spher, palm):
    acc_list = [0, 0]
    fails = [0, 0]
    for i, line in enumerate(test_data):
        # spher_pred = spher.predict(line)
        # palm_pred = palm.predict(line)
        # cyl_pred = cyl.predict(line)
        line = line.reshape(1, -1)
        predictions = [spher.predict(line),
                       palm.predict(line)]
        # print(predictions)

        pred_ind = predictions.index(max(predictions))

        if pred_ind == test_label[i]:
            acc_list[pred_ind] += 1
        else:
            fails[pred_ind] += 1
    # print(fails)

    return np.array(acc_list) / 150



if __name__ == '__main__':
    train_feat, _, names = load_train_data()
    test_feat, _ = load_test_data('Database_1_total_ch2')
    names = names[4], names[3]

    test_feat = np.concatenate((test_feat[600:750, :],
                                test_feat[450:600, :]), axis=0)
    test_label = np.concatenate((np.zeros((150, 1)), np.ones((150, 1))), axis=0)

    max_by_num = []
    n_list = list(range(200, 500, 10))
    for n in n_list:

        spher, palm = train_classifiers(train_feat, zero_num=n, alpha=0.001)

        acc_list = classify(test_feat, test_label, spher=spher, palm=palm)

        max_by_num.append(acc_list)

        # for i in range(3):
        #     print(f'Acc en {names[i]}: {acc_list[i]}')

    avg_by_num = [sum(acc)/2 for acc in max_by_num]
    ind_max_by_num = avg_by_num.index(max(avg_by_num))
    print(f'El maximo valor encontrado en: {n_list[ind_max_by_num]} con '
          f'{max_by_num[ind_max_by_num]}')



    tot_x = np.array(n_list)

    avg_y = np.array(avg_by_num)


    plt.plot(tot_x, avg_y)
    # plt.plot(tot_x, )
plt.show()
