## requires pySerial to be installed
import numpy as np
import serial
from scipy.fftpack import fft
import matplotlib.pyplot as plt

write_to_file_path = "output.txt"
ser = serial.Serial("COM4", baudrate=115200, timeout=1)


#Crear list de datos que lueego pase a graficarse..


while True:
    list_data = []
    while len(list_data) < 10000:
        line = ser.readline()
        data = str(line.decode('cp437')).strip('\r0\r\n')
        if not data.isnumeric():
            data = 0
        print(data)
        list_data.append(int(data))

    print(list_data)
    list_data = np.array(list_data)

    N = len(list_data)
    T = 1.0 / 800.0

    yf = fft(list_data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()
    plt.close()

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
