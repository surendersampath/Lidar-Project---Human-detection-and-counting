import sys
import numpy as np

import numpy as np
import pyqtgraph as pg

# Set white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Generate random points
n = 100000
print('Number of points: ' + str(n))
data = np.random.normal(size=(2, n))

# Create the main application instance
app = pg.mkQApp()

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
pos = [{'pos': data[:, i]} for i in range(n)]

now = pg.ptime.time()
print(pos[0])
scatter.setData(pos)
print("Plot time: {} sec".format(pg.ptime.time() - now))

# Gracefully exit the application
sys.exit(app.exec_())
