from random import randint
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
import pickle,sys
from sklearn.cluster import DBSCAN
import math
from sklearn.ensemble import RandomForestClassifier
import cv2

filename = 'rfcmodel'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
rfc = pickle.load(infile)
scannum = -1

#Opening a serialised object
filename = 'Pickled files/p1'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
scanlist = pickle.load(infile)
scan_shape = scanlist.shape
x_array = np.array([])
y_array = np.array([])


#-Data forming - End

colors = []
for x in range(20):
    colors.append('%06X' % randint(0, 0xFFFFFF))


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
        self.timer.setInterval(0) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.plotItem = self.addPlot(title='Postive \U0001f600')
        self.plotItem.getViewBox().setRange(xRange=(-3000,6000),yRange=(-6000,4000))

        self.plotDataItem = self.plotItem.plot([], pen=None,symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)



    def onNewData(self):

        global scannum
        self.plotItem.clear()
        humans = 0
        scannum = scannum + 1
        if scannum==scan_shape[0]-1:
            return
        if scannum==scan_shape[0]:
            print("End of scans")
            sys.exit()
        scan=newscan(scannum)

        centers = scan
        # Compute DBSCAN
        db = DBSCAN(eps=150, min_samples=15).fit(centers)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        label_list = set(labels)

        def getFeatures(clust_x, clust_y, clustersize):

            x_mean = sum(clust_x)/clustersize
            y_mean = sum(clust_y)/clustersize

            clust_x_sorted = np.sort(clust_x)
            clust_y_sorted = np.sort(clust_y)
            x_median = np.median(clust_x_sorted)
            y_median = np.median(clust_y_sorted)

            distance = math.sqrt(x_median*x_median + y_median*y_median)

            sum_std_diff = sum_med_diff = 0
            for i in range(clustersize):
                sum_std_diff += pow(clust_x[i]-x_mean, 2) + pow(clust_y[i]-y_mean, 2)
                sum_med_diff += math.sqrt(pow(clust_x[i]-x_median, 2)+pow(clust_y[i] - y_median, 2))

            std = math.sqrt(1/(clustersize-1)*sum_std_diff)
            avg_med_dev = sum_med_diff / clustersize

            first_elem = [clust_x[0],clust_y[0]]  #81
            last_elem  = [clust_x[-1],clust_y[-1]]  #82

            prev_ind = 0
            next_ind = 0
            prev_jump = 0
            next_jump = 0
            occluded_left = 0
            occluded_right = 0

            #122

            width = math.sqrt(pow(clust_x[0]-clust_x[-1], 2) + pow(clust_y[0]-clust_y[-1],2))  #125 - Width

            points=np.array((clustersize,2))

            points2=np.vstack((clust_x,clust_y))
            points2=np.transpose(points2)


            W = np.zeros((2,2), np.float64)
            w = np.zeros((2,2), np.float64)
            U = np.zeros((clustersize, 2), np.float64)

            V = np.zeros((2,2), np.float64)

            w,u,vt=cv2.SVDecomp(points2,W,U,V)
            rot_points = np.zeros((clustersize,2), np.float64)

            W[0,0]=w[0]
            W[1,1]=w[1]
            rot_points = np.matmul(u,W)


            linearity=0
            for i in range(clustersize):
                linearity += pow(rot_points[i, 1], 2)


            #Circularity
            A = np.zeros((clustersize,3), np.float64)
            B = np.zeros((clustersize,1), np.float64)


            for i in range(clustersize):
                A[i,0]=-2.0 * clust_x[i]
                A[i,1]=-2.0 * clust_y[i]
                A[i,2]=1
                B[i,0]=math.pow(clust_x[i], 2)-math.pow(clust_y[i], 2)

            sol = np.zeros((3,1),np.float64)
            cv2.solve(A, B, sol, cv2.DECOMP_SVD)

            xc = sol[0,0]
            yc = sol[1,0]
            rc = math.sqrt(pow(xc, 2)+pow(yc, 2)) - sol[2,0]


            circularity = 0
            for i in range(clustersize):
                circularity += pow(rc - math.sqrt(pow(xc - clust_x[i], 2) + pow(yc-clust_y[i], 2)), 2)


            radius = rc #Radius

            mean_curvature = 0  #207 Mean_Curvature

            boundary_length = 0  #Boundary_Length
            last_boundary_seg = 0 #Boundary_Length
            boundary_regularity = 0
            sum_boundary_reg_sq = 0

            #Mean Angualar Difference

            left = 2
            mid = 1
            right=0

            ang_diff=0

            while(left!=clustersize):
                mlx  =  clust_x[left] - clust_x[mid]
                mly  =  clust_y[left] - clust_y[mid]
                L_ml =  math.sqrt(mlx*mlx + mly*mly)

                mrx  = clust_x[right] - clust_x[mid]
                mry  = clust_y[right] - clust_y[mid]
                L_mr = math.sqrt(mrx * mrx + mry * mry)

                lrx  = clust_x[left] - clust_x[right]
                lry  = clust_y[left] - clust_y[right]
                L_lr = math.sqrt(lrx * lrx + lry * lry)


                boundary_length+= L_mr
                sum_boundary_reg_sq += L_mr*L_mr
                last_boundary_seg = L_ml

                A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
                B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

                th = math.atan2(B,A)

                if th<0:
                    th += 2*math.pi

                ang_diff += th/clustersize

                s = 0.5 * (L_ml + L_mr + L_lr)
                area = math.sqrt(s * (s - L_ml) * (s - L_mr) * (s - L_lr))


                if th>0:
                    mean_curvature += 4 * (area) / (L_ml * L_mr * L_lr * clustersize)
                else:
                    mean_curvature -= 4 * (area) / (L_ml * L_mr * L_lr * clustersize)


                left=left+1
                mid=mid+1
                right=right+1  #While loop ends


            boundary_length += last_boundary_seg
            sum_boundary_reg_sq += last_boundary_seg*last_boundary_seg



            boundary_regularity = math.sqrt((sum_boundary_reg_sq - math.pow(boundary_length, 2) / clustersize)/(clustersize - 1))



            #Mean Angular difference
            first = 0
            mid   = 1
            last  = -1


            sum_iav = 0
            sum_iav_sq = 0



            while(mid < clustersize-1):
                mlx = clust_x[first] -clust_x[mid]
                mly = clust_y[first] -clust_y[mid]

                mrx  = clust_x[last]-clust_x[mid]
                mry  = clust_y[last]-clust_y[mid]
                L_mr = math.sqrt(mrx * mrx + mry * mry)

                A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
                B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

                th = math.atan2(B, A)

                if(th<0):
                    th += 2 * math.pi


                sum_iav += th

                sum_iav_sq += th*th

                mid = mid+1


            iav = sum_iav/clustersize
            std_iav = math.sqrt((sum_iav_sq - pow(sum_iav, 2) / clustersize) / (clustersize - 1))




            features=[clustersize, std, avg_med_dev, width, linearity, circularity,
                                radius, boundary_regularity, mean_curvature, ang_diff, iav, std_iav,
                                distance, distance/clustersize]
            return features

        for label in label_list:

            index = labels == label
            cluster = scan[index]

            clus_max_x = max(cluster[:, 0]) + 200
            clus_min_x = min(cluster[:, 0]) - 200
            clus_max_y = max(cluster[:, 1]) + 200
            clus_min_y = min(cluster[:, 1]) - 200

            features = getFeatures(cluster[:, 0], cluster[:, 1], cluster.shape[0])

            c1 = self.plotItem.plot(cluster[:, 0], cluster[:, 1], symbol='o', symbolPen=colors[label], symbolSize=8)
            txt = '\U0001f600'
            tx1 = 'total humans detected =' + str(humans)
            # text = pg.TextItem(html=txt, anchor=(0, 0), border='w', fill=(0, 0, 255, 100))
            text = pg.TextItem(html=txt, anchor=(0, 0), border='w', fill=None)

            if (rfc.predict([features])==1):
                humans += 1
                clus_max_x = max(cluster[:, 0])
                clus_min_x = min(cluster[:, 0])
                clus_max_y = max(cluster[:, 1])
                clus_min_y = min(cluster[:, 1])
                x1 = [clus_min_x ,clus_min_x, clus_max_x, clus_max_x, clus_min_x]
                y1 = [clus_min_y ,clus_max_y, clus_max_y, clus_min_y, clus_min_y]
                self.plotItem.plot(x1, y1, pen='r')
                self.plotItem.addItem(text)
                text.setPos(clus_max_x, clus_max_y)
                print(clus_max_x,clus_max_y)
                print('Humans in frame', humans)
def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=False) # True seems to work as well
    pg.setConfigOption('background', 'w')

    win = MyWidget()
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()



if __name__ == "__main__":
    main()
