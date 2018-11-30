import numpy as np
from functools import reduce
from sklearn.preprocessing import Normalizer
import serial
from time import sleep

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
    if pulses == 0:
        return True
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
    if len(data) == 1:
        data = data.reshape(1, -1)
        return scaler.transform(data)
    if len(data) > 1:
        return scaler.transform(data)


def get_measure(length):
    test_feat = []
    print('Tomando medicion !!')
    while len(test_feat) < length:  #esa cantidad mas 1 del label
        data = read_line()
        test_feat.append(float(data))
    return np.array(test_feat).reshape(1, -1)


ser = serial.Serial("COM4", baudrate=115200, timeout=1)

data_len = 1200
count = 0
features = np.zeros((1, data_len))

while count < 300:

    print(f"Ingresando dato {count} a la matriz.. preparese")
    label = 3 # int(input('Ingrese label de la muestra proxima: '))
    while True:
        if debouncer(pulses=0):
            test_data = get_measure(data_len)
            test_data = np.insert(test_data, 0, label, axis=1)
            if not features.any():
                features = test_data
            else:
                features = np.append(features, test_data, axis=0)
            break
    print('###  Fin de la muestra  ###\n')
    count += 1

save = input('desea guardar los datos recopilados? [y/n]: ')
if save.lower() == 'y':
    print(features.shape)
    features = normalizar(features)
    save_name = input('Ingrese nombre del archivo: ')
    np.save(save_name, features)
