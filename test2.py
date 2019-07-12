

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(800,800)
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')


w1 = view.addPlot()  #p6=w1


n = 2
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
pos = np.random.normal(size=(2,n), scale=1e-5)
dos = np.random.normal(size=(4,n), scale=1e-5)

spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
dots = [{'pos': dos[:,i], 'data': 1} for i in range(2)] + [{'pos': [0,0], 'data': 1}]

s1.addPoints(spots)
s1.clear()
s1.setData(dots)

w1.addItem(s1)




if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
