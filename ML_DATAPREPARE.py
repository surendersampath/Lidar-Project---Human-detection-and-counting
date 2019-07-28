import pandas as pd
import numpy as np
import pickle
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

k = 0

filepath="Pickled files/n1.csv"
def parsecoord(coortxt):
    coortxt=coortxt[1:-2]
    xy=coortxt.split(';')
    x=float(xy[0])
    y=float(xy[1])
    return x,y
with open(filepath,"r") as file:
    line=file.readline()
    print(k)
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
        print('1')
        print('2.....')
        print('3........')
        cls()
        k = k + 1

filename = 'n1'
outfile = open(filename,'wb')
pickle.dump(scanlist,outfile)
print('Data prepare complete')

print(scanlist.shape)


