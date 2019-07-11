
import numpy as np
import pickle,sys

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ###########################setting up array##################################################
cannum = -1



#Opening a serialised object
filename = 'scanlist'
infile = open(filename,'rb')
scanlist = pickle.load(infile)
print(scanlist.shape)
data = scanlist.reshape(3679560, 2)
print(data.shape)
scanlist_new = scanlist[0:1000,:]
print(scanlist_new.shape)
# ###########################set up array complete   ##########################################
print(scanlist_new[0])
f1 = 0


def calc_plt(labels):
    global f1
    f1=1
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
            plt.pause(0.02)

            f1 = 1
def calc_lablel(data):
    centers = data
    X, labels_true = make_blobs(n_samples=1080, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    calc_plt(labels)


for i in range(scanlist_new.shape[0]):
    print('flag STATUS', f1)
    print('scannum',i)
    data = scanlist_new[i].reshape(1080,2)

    centers = data
    X, labels_true = make_blobs(n_samples=1080, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    calc_lablel(data)
    if f1==1:
        plt.clf()

plt.title('fig1')


plt.show()





