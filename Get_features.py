import numpy as np
import serial
from time import sleep


ser = serial.Serial("COM4", baudrate=115200, timeout=1)
count = 0

features = []
labels = []
while count < 60:

    print(f"Ingresando dato {count} a la matriz.. preparese")
    feature_line = [input('Ingrese label de la muestra proxima: ')]
    print('Tomando muestra en 3..')
    sleep(0.5)
    print('Tomando muestra en 2..')
    sleep(0.5)
    print('Tomando muestra en 1..')
    sleep(0.5)
    print('¡ YA !')
    sleep(0.5)
    while len(feature_line) <= 5000:  #esa cantidad mas 1 del label
        line = ser.readline()
        data = str(line.decode('cp437')).strip('\r0\r\n')
        if not data.isnumeric():
            data = 0
        feature_line.append(int(data))
    print(feature_line)
    print('###  Fin de la muestra  ###')
    features.append(feature_line)
    count += 1
    print()

save = input('desea guardar los datos recopilados? [y/n]: ')
if save.lower() == 'y':
    features = np.array(features)
    save_name = input('Ingrese nombre del sector medido: ')
    np.save(f"features_{save_name}", features)
