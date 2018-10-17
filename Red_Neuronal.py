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

# ser = serial.Serial("COM4", baudrate=115200, timeout=1)


def read_line():
    line = ser.readline()
    data = str(line.decode('cp437')).strip('\r0\r\n')
    if not data.isnumeric():
        data = 0
    return float(data)


def debouncer(pulses=5):
    test = []
    for _ in range(pulses):
        if read_line():
            test.append(True)
        else:
            test.append(False)
    eval = reduce(lambda x, y: x and y, test)
    return eval


#Cambiar normalizacion o.. normalizar los datos de entrenamiento tambien
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
    feats = normalizar(feats)
    return feats, labels, _names

def load_test_data(path):
    data = np.load(path + '.npy')
    labels, feats = data[:, :1], data[:, 1:2501]
    feats = normalizar(feats)
    return labels, feats


def get_measure(length):
    test_feat = []
    print('Tomando medicion !!')
    while len(test_feat) < length:  #esa cantidad mas 1 del label
        data = read_line()
        test_feat.append(float(data))
    return normalizar(np.array(test_feat))


if __name__ == '__main__':

    train_data, train_labels, names = load_train_data()

    plot_data = []

    for neuron_num in (64, 128, 256, 512):

        with tf.device('/device:GPU:1'):
            model = keras.Sequential([
            keras.layers.Dense(neuron_num, activation=tf.nn.relu),
            keras.layers.Dense(6, activation=tf.nn.softmax)])

            model.compile(optimizer=tf.train.AdamOptimizer(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_data, train_labels, epochs=17)

        test_mode = 'dataset'   # input('Modo de prueba [sensor/dataset]: ')

        if test_mode.lower() == 'sensor':
            continuar = True
            while continuar:
                #Una fila de datos
                test_data = np.zeros((1, 2500))
                while not debouncer(pulses=3):
                    test_data = get_measure(2500)
                    test_data = normalizar(test_data)[0]
                # Add the image to a batch where it's the only member.
                test_data = (np.expand_dims(test_data, 0))
                if test_data.any():

                    pred_single = model.predict(test_data)

                    ind = np.argmax(pred_single)
                    print(pred_single)
                    print(names[ind] + '\n')


                    continuar = True if input('Desea continuar? y/n: ') == 'y' else False

        if test_mode.lower() == 'dataset':
            Tlabels, Tfeats = load_test_data('Database_1_total_ch2')
            print(Tfeats.shape)
            test_loss, test_acc = model.evaluate(Tfeats, Tlabels)

            print('Test accuracy:', test_acc)

            plot_data.append((neuron_num, test_acc))

    plt_x = [x[0] for x in plot_data]
    plt_y = [x[1] for x in plot_data]

    plt.plot(plt_x, plt_y)
    plt.show()


