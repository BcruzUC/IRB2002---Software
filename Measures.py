## requires pySerial to be installed
import numpy as np
import serial
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy import signal
import scipy as sp
from collections import defaultdict
import scipy.io as sio
import os



# write_to_file_path = "output.txt"
# ser = serial.Serial("COM4", baudrate=115200, timeout=1)
#
#
# #Crear list de datos que lueego pase a graficarse..
#
#
# while True:
#     list_data = []
#     while len(list_data) < 10000:
#         line = ser.readline()
#         data = str(line.decode('cp437')).strip('\r0\r\n')
#         if not data.isnumeric():
#             data = 0
#         print(data)
#         list_data.append(int(data))
#
#     print(list_data)
#     list_data = np.array(list_data)
#
#     N = len(list_data)
#     T = 1.0 / 800.0
#
#     yf = fft(list_data)
#     xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#     plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#     plt.grid()
#     plt.show()
#     plt.close()

#### Archivo original ! ###

# output_file = open(write_to_file_path, "w");
# while True:
#     line = ser.readline();
#     line = line.decode("utf-8") #ser.readline returns a binary, convert to string
#     print(line);
#     output_file.write(line);



# Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()

## CHANGE DATABASE 1 & 2 TO TOTAL FORM


def sort_measures(data, channel='ch1'):
    data_keys = [name for name in list(data) if channel in name]
    total_data = np.zeros((300, 2501))
    for i, name in enumerate(data_keys):
        y, x = data[name].shape
        label = np.ones((y, 1)) * i
        temp_data = np.concatenate((label, data[name]), axis=1)
        if not total_data.any():
            total_data[:, :] = temp_data
        else:
            total_data = np.concatenate((total_data, temp_data), axis=0)
    return total_data, np.array(data_keys)


temp_dict = defaultdict(list)
folder_name = 'Database_2'

dir_list = os.listdir(folder_name)
for path in dir_list:
    data = sio.loadmat('{}/{}'.format(folder_name, path))
    dict_keys = list(data)[3:]
    for _key in dict_keys:
        _array = np.array(data[_key])
        temp_dict[_key].append(_array)



total_dict = {}
for ind in list(temp_dict):
    test = np.array(temp_dict[ind])
    conct = np.concatenate((test[0, :, :], test[1, :, :], test[2, :, :]), axis=0)
    if folder_name == 'Database_1':
        conct = np.concatenate((test[0, :, :], test[1, :, :], test[2, :, :],
                            test[3, :, :], test[4, :, :]), axis=0)
    total_dict[ind] = conct

channel = input('Elija set de datos [ch1/ ch2]: ')
save_data, data_keys = sort_measures(total_dict, channel=channel)
save_data = abs(save_data)

# np.save('{}_total_{}'.format(folder_name, channel), save_data)
np.save('{}_names_{}'.format(folder_name, channel), data_keys)

