
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

class MyWidget(pg.GraphicsWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(2000) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.plotItem = self.addPlot(title="Lidar points")

        self.plotDataItem = self.plotItem.plot([], pen=None,
            symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)


    def setData(self, x, y):
        self.plotDataItem.setData(x, y)


    def onNewData(self):
        numPoints = 1
        x = np.random.normal(size=numPoints)
        y = np.random.normal(size=numPoints)
        self.setData(x, y)


def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=False) # True seems to work as well

    win = MyWidget()
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()

if __name__ == "__main__":
    main()
