
import numpy as np
import pickle,sys
from random import randint

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
# ###########################setting up array##################################################
cannum = -1
#Opening a serialised object
filename = 'scanlist'
infile = open(filename,'rb')
scanlist = pickle.load(infile)
print(scanlist.shape)
# ###########################set up array complete   ##########################################
print(scanlist.shape[0])






# Generate sample data
centers = scanlist[0]

# Compute DBSCAN
db = DBSCAN(eps=100, min_samples=10).fit(centers)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print(set(labels))
# #############################################################################
label_list=set(labels)


colors = []
for x in range(len(set(labels))):
    colors.append('%06X' % randint(0, 0xFFFFFF))

print(colors)

count = 0



print(count)

for label in label_list:
    index=labels==label
    cluster=scan[index]
    print(cluster.shape)
