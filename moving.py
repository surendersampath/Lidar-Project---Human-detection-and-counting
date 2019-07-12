from random import randint
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pickle,sys
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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

colors = []

for x in range(100):
    colors.append('%06X' % randint(0, 0xFFFFFF))


print(scanlist.flatten())

scanlist_flat = scanlist.flatten()


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
        self.timer.setInterval(5) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.plotItem = self.addPlot(title="Lidar points")
        self.plotItem.getViewBox().setRange(xRange=(-3000,6000),yRange=(-6000,4000))

        self.plotDataItem = self.plotItem.plot([], pen=None,symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)


    def setData(self, x, y):



        # self.plotDataItem.setData(x, y,pen='r')
        return



    def onNewData(self):

        global scannum
        self.plotItem.clear()
        scannum = scannum + 1
        if scannum==scan_shape[0]-1:
            return
        if scannum==scan_shape[0]:
            print("End of scans")
            sys.exit()
        scan=newscan(scannum)

        centers = scan
        # Compute DBSCAN
        db = DBSCAN(eps=100, min_samples=10).fit(centers)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        label_list = set(labels)




        for label in label_list:
            index = labels == label
            cluster = scan[index]
            # print(cluster.shape)
            c1 = self.plotItem.plot(cluster[:, 0], cluster[:, 1], symbol='x', symbolPen=colors[label], name='red')

        # self.setData(scan[:,0], scan[:,1])

        # self.plotItem.setData(scan,pen='r')




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
