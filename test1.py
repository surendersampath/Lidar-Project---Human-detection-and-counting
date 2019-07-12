

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
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






#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
win.setRange(xRange=[-2000,10000],yRange=[-8500,6000])
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)






p6 = win.addPlot(title="Updating plot")  #w1
p6.setRange(xRange=[-3000,6000],yRange=[-6000,4000])
s1 = pg.ScatterPlotItem(pen='y')
# s1.setRange(xRange=[-3000,6000],yRange=[-6000,4000])

# data = np.random.normal(size=(10,1000))
# ptr = 0


def newscan(scannum):
    print('scan number : ',scannum)
    if(scannum <= scan_shape[0]):

        return scanlist[scannum]



def update():
    # global curve, data, ptr, p6
    # # curve.setData(data[ptr%10])
    # if ptr == 0:
    #     p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    # ptr += 1
    global scannum
    scannum = scannum + 1
    if scannum==scan_shape[0]-1:
        return
    if scannum==scan_shape[0]:
        print("End of scans")
        sys.exit()
    scan=newscan(scannum)
    # curve.setData(scan[:, 0], scan[:, 1])
    s1.setData(scan[:, 0], scan[:, 1])

    # p6.addItem(curve)


#TIMER FUNCS
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)








## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
