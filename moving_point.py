from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pickle,sys


scannum = -1

#Opening a serialised object
filename = 'scanlist'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
scanlist = pickle.load(infile)

print(scanlist.shape)

scan_shape = scanlist.shape

print(type(scan_shape))

x_array = np.array([])
y_array = np.array([])


#-Data forming - End






def newscan(scannum):
    print('scan number : ',scannum)
    if(scannum <= scan_shape[0]):
        return scanlist[scannum]






class MyWidget(pg.GraphicsWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        # self.timer.setInterval(5) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.plotItem = self.addPlot(title="Lidar points")
        self.plotItem.getViewBox().setRange(xRange=(-3000,6000),yRange=(-6000,4000))

        self.plotDataItem = self.plotItem.plot([], pen=None,symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)


    def setData(self, x, y):

        print(x.shape,y.shape)

        for z in range(len(x)):
            A = np.array([x[z]])
            B = np.array([y[z]])
            # print(A,B)
            self.plotDataItem.setData(A, B,pen='r')




    def onNewData(self):

        global scannum
        scannum = scannum + 1
        if scannum==scan_shape[0]-1:
            return
        if scannum==scan_shape[0]:
            print("End of scans")
            sys.exit()
        scan=newscan(scannum)
        self.setData(scan[:,0], scan[:,1])




def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=False) # True seems to work as well
    pg.setConfigOption('background', 'w')

    win = MyWidget()
    # win.setRange(xRange=(-2000,10000),yRange=(-8500,6000))
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()



if __name__ == "__main__":
    main()
