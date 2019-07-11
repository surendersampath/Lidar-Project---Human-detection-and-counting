
import numpy as np
import pickle,sys

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
data = scanlist.reshape(3679560, 2)
print(data.shape)
scanlist_new = scanlist[0:3,:]
print(scanlist_new.shape)
# ###########################set up array complete   ##########################################
print(scanlist[0])
for x in range(3):
    print(scanlist_new[x])
    data = scanlist_new[x].reshape(1080,2)
    centers = data
    X, labels_true = make_blobs(n_samples=1080, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

# ##################h###########################################################
# Generate sample data


# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    plt.pause(1)
    plt.clf()

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


