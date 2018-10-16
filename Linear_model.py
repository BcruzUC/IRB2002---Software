from sklearn import linear_model as lm
from sklearn.preprocessing import Normalizer
import numpy as np
import random as rn
from collections import defaultdict



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


def train_classifiers(feats):
    y = np.ones((600, 1))    #Evaluar valores en los test
    spher_class = lm.Ridge(alpha=0.001, normalize=True)
    palm_class = lm.Ridge(alpha=0.001, normalize=True)
    cyl_class = lm.Ridge(alpha=0.001, normalize=True)

    spher_class.fit(feats[3000:3600, :], y)  #clase 5
    palm_class.fit(feats[2400:3000, :], y)   #clase 4
    cyl_class.fit(feats[:600, :], y)         #clase 1

    return spher_class, palm_class, cyl_class

def classify(test_data, test_label, spher, palm, cyl):
    acc_list = [0, 0, 0]
    fails = [0, 0, 0]
    for i, line in enumerate(test_data):
        # spher_pred = spher.predict(line)
        # palm_pred = palm.predict(line)
        # cyl_pred = cyl.predict(line)
        line = line.reshape(1, -1)
        print(line)
        predictions = [spher.predict(line),
                       palm.predict(line),
                       cyl.predict(line)]
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
    names = names[4], names[3], names[0]

    test_feat = np.concatenate((test_feat[600:750, :],
                                test_feat[450:600, :],
                                test_feat[:150, :]), axis=0)
    test_label = np.concatenate((np.zeros((150, 1)), np.ones((150, 1)),
                                 np.ones((150, 1)) * 2), axis=0)

    spher, palm, cyl = train_classifiers(train_feat)

    acc_list = classify(test_feat, test_label, spher=spher, palm=palm, cyl=cyl)

    for i in range(3):
        print(f'Acc en {names[i]}: {acc_list[i]}')

    asd = np.ones((1, 2500)) * 0.3
    print(spher.predict(asd.reshape(1, -1)))



