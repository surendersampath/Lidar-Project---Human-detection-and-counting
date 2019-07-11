from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pickle

scannum = -1



#-Data forming - start
filepath="d1_sorted.csv"
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

filename = 'scanlist'
outfile = open(filename,'wb')
pickle.dump(scanlist,outfile)
