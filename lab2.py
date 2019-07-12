import sys
import pickle,sys

import numpy as np
import pyqtgraph as pg

# Set white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
global scannum

# Generate random points
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

# Create the main application instance
app = pg.mkQApp()

def newscan(scannum):
    print('scan number : ',scannum)
    if(scannum <= scan_shape[0]):
        return scanlist[scannum]


# Create the view
view = pg.PlotWidget()
view.resize(800, 600)
view.setWindowTitle('Scatter plot using pyqtgraph with PyQT5')
view.setAspectLocked(True)
view.show()

# Create the scatter plot and add it to the view
scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
view.addItem(scatter)

# Convert data array into a list of dictionaries with the x,y-coordinates



now = pg.ptime.time()


while(scan_shape[0]!=scannum):

    scannum=scannum+1

    if scannum == scan_shape[0]:
        print("End of scans")
        sys.exit()
    scan = newscan(scannum)
    scatter.setData(scan[:,0], scan[:,1])
    view.show()

# def scatplot():


    # scatter.setData(pos)





print("Plot time: {} sec".format(pg.ptime.time() - now))

# Gracefully exit the application
sys.exit(app.exec_())
