import math, time, serial
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time

win = pg.GraphicsWindow()
win.setWindowTitle('pyqtgraph example: Scrolling Plots')

p1 = win.addPlot()
p2 = win.addPlot()

curve1 = p1.plot()
curve2 = p2.plot()
data1 = np.empty(500)
data2 = np.empty(500)

# instanciacao dos dados
Mpu1_xF = []
Mpu1_yF = []

# configuracao porta serial
ser = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=19200,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.SEVENBITS
)


def dist(x, y):
    return math.sqrt((x * x) + (y * y))


def angle_accel_y(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)


def angle_accel_x(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


# def update1():
#	global Mpu1_x,data1
#	print "dato update", Mpu1_x
#	data1[-1] = Mpu1_x
#	curve1.setData(data1)

# update all plots
def update():
    update1()
    # update2()
    # update3()


timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)


class Kalman:

    def __init__(self):
        self.Q_angle = 0.001
        self.Q_bias = 0.003
        self.R_measure = 3 / 100.

        self.angle = 0.0
        self.bias = 0.0
        self.rate = 0.0
        self.tempo = 0.0

        self.p = np.matrix([[0., 0.], [0., 0.]])
        self.f = np.matrix([[1., -self.tempo], [0., 1.]])
        self.q = np.matrix([[self.Q_angle, 0.], [0., self.Q_bias]])

    # self.k = np.matrix([[0.0],[0.0]])

    def filtro(self, angle, gyro, tempo):
        self.rate = gyro - self.bias
        self.angle += (tempo * gyro)
        self.p = np.dot(np.dot(self.f, self.p), np.transpose(self.f)) + self.q
        s = self.p[0, 0] + self.R_measure
        self.k = np.matrix([[self.p[0, 0] / s], [self.p[1, 0] / s]])
        y = angle - self.angle
        self.angle += (self.k[0, 0] * y)
        self.bias += (self.k[1, 0] * y)

        self.p[0, 0] = (-self.k[0, 0] * self.p[0, 0])
        self.p[0, 1] = (-self.k[0, 0] * self.p[0, 1])
        self.p[1, 0] = (-self.k[1, 0] * self.p[0, 0])
        self.p[1, 1] = (-self.k[1, 0] * self.p[0, 1])
        # print self.p

        return self.angle


kalman_x = Kalman()
kalman_y = Kalman()


def update1():
    global data1
    # while True:

    while (ser.inWaiting() == 0):
        pass

    datos = ser.readline()
    # print "datos",datos
    splitedArray = [s for s in datos.split(":")]
    temp = float(splitedArray[1])
    # print "tempo",temp
    ac_x = float(splitedArray[2])
    # print "acx", ac_x
    ac_y = float(splitedArray[3])
    # print "acy", ac_y
    ac_z = float(splitedArray[4])
    # print "acz", ac_z
    go_x = float(splitedArray[5])
    go_y = float(splitedArray[6])
    go_z = float(splitedArray[7])

    angle_x = angle_accel_x(ac_x, ac_y, ac_z)
    # print "angulo x", angle_x
    angle_y = angle_accel_y(ac_x, ac_y, ac_z)
    # print "angulo y", angle_y

    Mpu1_x = kalman_x.filtro(angle_x, go_x, temp)
    print
    "angulo x kalman", Mpu1_x
    Mpu1_y = kalman_y.filtro(angle_y, go_y, temp)
    print
    "angulo y kalman", Mpu1_y

    # Mpu1_xF.append(Mpu1_x)
    # Mpu1_yF.append(Mpu1_y)

    data1[:-1] = data1[1:]
    data1[-1] = Mpu1_x
    curve1.setData(data1)


    data2[:-1] = data2[1:]
    data2[-1] = Mpu1_y
    curve2.setData(data2)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
