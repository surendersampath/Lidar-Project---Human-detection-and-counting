import sys
import pyqtgraph as pg
import numpy as np
from PyQt5 import *



# Set white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Generate random points
# n = 1
# print('Number of points: ' + str(n))
# data = np.random.normal(size=(2, n))

# Create the main application instance
app = pg.mkQApp()

# Create the view
view = pg.PlotWidget()
view.resize(600, 800)
view.setWindowTitle('Scatter plot using pyqtgraph with PyQT5 LIDAR DATA')
view.setAspectLocked(True)
view.show()

# Create the scatter plot and add it to the view
scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
view.addItem(scatter)

# Convert data array into a list of dictionaries with the x,y-coordinates
# pos = [{'pos': data[:, i]} for i in range(n)]
pos = []
print(type(pos))
now = pg.ptime.time()
# scatter.setData(pos)

# Getting the points part -


filepath="d1_test.csv"
def parsecoord(coortxt):
    coortxt=coortxt[1:-2]
    xy=coortxt.split(';')
    x=float(xy[0])
    y=float(xy[1])
    return x,y
with open(filepath,"r") as file:
    line=file.readline()
    scanlist=np.array([]).reshape(0,1080,2)
    while line:
        ll=line.split(',')
        ll=ll[1:-1]
        nump=len(ll)
#         print(nump) 1080
        scan=np.array([]).reshape(0,2)
        for i in range(nump):
            coortex=ll[i]
            x,y=parsecoord(coortex)
            xy=[[x,y]]
            scan=np.append(scan,xy,axis=0)
        line=file.readline()
        scan=np.reshape(scan,(1,1080,2))
        scanlist=np.append(scanlist,scan,axis=0)
#
print(pos)

print(scanlist.shape)

scan_shape = scanlist.shape

print(type(scan_shape))

x_array = np.array([])
y_array = np.array([])

dict1 = {'pos': scanlist[0][1]}
for scan in range(scan_shape[0]):
    for element in range(scan_shape[1]):
        dict1 = {'pos': scanlist[0][element]}
        x_array = np.append(x_array,scanlist[scan][element][0])
        y_array = np.append(y_array,scanlist[scan][element][1])
        pos.append(dict1)
        # print(len(pos))
print(len(x_array))
print(len(x_array))

# scatter.setData(pos)



# Getting the points part - End




print("Plot time: {} sec".format(pg.ptime.time() - now))

# Gracefully exit the application
sys.exit(app.exec_())

print('end of app')



