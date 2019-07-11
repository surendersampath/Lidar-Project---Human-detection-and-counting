
import numpy as np
import pickle,sys
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pickle,sys




# ###########################setting up array##################################################
cannum = -1
#Opening a serialised object
filename = 'scanlist'
infile = open(filename,'rb')
scanlist = pickle.load(infile)
print(scanlist.shape)
data = scanlist.reshape(scanlist.shape[0]*scanlist.shape[1], 2)
print(data.shape)
scanlist_new = scanlist[0:3,:]
print(scanlist_new.shape)
# ###########################set up array complete   ##########################################
#Counter for scannum
scannum = 0
#Comment below line to process the full data set
scanlist = scanlist_new
#Create a label list array
labellist = np.zeros(shape=(1,1080))



def getLabel(data):
    data = data.reshape(1080, 2)
    centers = data
    X, labels_true = make_blobs(n_samples=1080, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return labels


for x in range(scanlist.shape[0]):

    data = scanlist[scannum]
    label_data = getLabel(data)
    index0=label_data==0
    pts0=data[index0]
    print(np.unique(label_data))
    print(label_data.shape)





    scannum = scannum+1







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
        self.plotItem.getViewBox().setRange(xRange=(-2000,10000),yRange=(-8500,6000))

        self.plotDataItem = self.plotItem.plot([], pen=None,symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)


    def setData(self, x, y):
        self.plotDataItem.setData(x, y)




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

    win = MyWidget()
    # win.setRange(xRange=(-2000,10000),yRange=(-8500,6000))
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()





if __name__ == "__main__":
    main()




