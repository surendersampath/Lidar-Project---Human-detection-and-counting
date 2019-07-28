import numpy as np
import pickle
from sklearn.cluster import DBSCAN
import math
import cv2
import time
start_time = time.time()



featureslist = np.array([])


# Returns features
def getFeatures(clust_x, clust_y, clustersize):
    x_mean = sum(clust_x) / clustersize
    y_mean = sum(clust_y) / clustersize

    clust_x_sorted = np.sort(clust_x)
    clust_y_sorted = np.sort(clust_y)
    x_median = np.median(clust_x_sorted)
    y_median = np.median(clust_y_sorted)

    distance = math.sqrt(x_median * x_median + y_median * y_median)

    sum_std_diff = sum_med_diff = 0
    for i in range(clustersize):
        sum_std_diff += pow(clust_x[i] - x_mean, 2) + pow(clust_y[i] - y_mean, 2)
        sum_med_diff += math.sqrt(pow(clust_x[i] - x_median, 2) + pow(clust_y[i] - y_median, 2))

    std = math.sqrt(1 / (clustersize - 1) * sum_std_diff)
    avg_med_dev = sum_med_diff / clustersize

    first_elem = [clust_x[0], clust_y[0]]  # 81
    last_elem = [clust_x[-1], clust_y[-1]]  # 82

    prev_ind = 0
    next_ind = 0
    prev_jump = 0
    next_jump = 0
    occluded_left = 0
    occluded_right = 0

    # 122

    width = math.sqrt(pow(clust_x[0] - clust_x[-1], 2) + pow(clust_y[0] - clust_y[-1], 2))  # 125 - Width

    points = np.array((clustersize, 2))

    points2 = np.vstack((clust_x, clust_y))
    print('ff', points2.shape)
    points2 = np.transpose(points2)

    print('ff', points2.shape)

    W = np.zeros((2, 2), np.float64)
    w = np.zeros((2, 2), np.float64)
    U = np.zeros((clustersize, 2), np.float64)

    V = np.zeros((2, 2), np.float64)

    w, u, vt = cv2.SVDecomp(points2, W, U, V)
    print('ww', w, W)
    rot_points = np.zeros((clustersize, 2), np.float64)

    W[0, 0] = w[0]
    W[1, 1] = w[1]
    rot_points = np.matmul(u, W)
    # print(w)
    # print(u)
    # print(vt)
    # print(rot_points.shape)

    linearity = 0
    for i in range(clustersize):
        linearity += pow(rot_points[i, 1], 2)

    # Circularity
    A = np.zeros((clustersize, 3), np.float64)
    B = np.zeros((clustersize, 1), np.float64)

    for i in range(clustersize):
        A[i, 0] = -2.0 * clust_x[i]
        A[i, 1] = -2.0 * clust_y[i]
        A[i, 2] = 1
        B[i, 0] = math.pow(clust_x[i], 2) - math.pow(clust_y[i], 2)

    sol = np.zeros((3, 1), np.float64)
    cv2.solve(A, B, sol, cv2.DECOMP_SVD)

    xc = sol[0, 0]
    yc = sol[1, 0]
    rc = math.sqrt(pow(xc, 2) + pow(yc, 2)) - sol[2, 0]

    circularity = 0
    for i in range(clustersize):
        circularity += pow(rc - math.sqrt(pow(xc - clust_x[i], 2) + pow(yc - clust_y[i], 2)), 2)

    radius = rc  # Radius

    mean_curvature = 0  # 207 Mean_Curvature

    boundary_length = 0  # Boundary_Length
    last_boundary_seg = 0  # Boundary_Length
    boundary_regularity = 0
    sum_boundary_reg_sq = 0

    # Mean Angualar Difference

    left = 2
    mid = 1
    right = 0

    ang_diff = 0

    while (left != clustersize):
        mlx = clust_x[left] - clust_x[mid]
        mly = clust_y[left] - clust_y[mid]
        L_ml = math.sqrt(mlx * mlx + mly * mly)

        mrx = clust_x[right] - clust_x[mid]
        mry = clust_y[right] - clust_y[mid]
        L_mr = math.sqrt(mrx * mrx + mry * mry)

        lrx = clust_x[left] - clust_x[right]
        lry = clust_y[left] - clust_y[right]
        L_lr = math.sqrt(lrx * lrx + lry * lry)

        boundary_length += L_mr
        sum_boundary_reg_sq += L_mr * L_mr
        last_boundary_seg = L_ml

        A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
        B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

        th = math.atan2(B, A)

        if th < 0:
            th += 2 * math.pi

        ang_diff += th / clustersize

        s = 0.5 * (L_ml + L_mr + L_lr)
        area = math.sqrt(s * (s - L_ml) * (s - L_mr) * (s - L_lr))

        if th > 0:
            mean_curvature += 4 * (area) / (L_ml * L_mr * L_lr * clustersize)
        else:
            mean_curvature -= 4 * (area) / (L_ml * L_mr * L_lr * clustersize)

        left = left + 1
        mid = mid + 1
        right = right + 1  # While loop ends

    boundary_length += last_boundary_seg
    sum_boundary_reg_sq += last_boundary_seg * last_boundary_seg

    boundary_regularity = math.sqrt(
        (sum_boundary_reg_sq - math.pow(boundary_length, 2) / clustersize) / (clustersize - 1))

    # Mean Angular difference
    first = 0
    mid = 1
    last = -1

    sum_iav = 0
    sum_iav_sq = 0

    while (mid != clustersize):
        mlx = clust_x[first] - clust_x[mid]
        mly = clust_y[first] - clust_y[mid]

        mrx = clust_x[last] - clust_x[mid]
        mry = clust_y[last] - clust_y[mid]
        L_mr = math.sqrt(mrx * mrx + mry * mry)

        A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
        B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

        th = math.atan2(B, A)

        if (th < 0):
            th += 2 * math.pi

        sum_iav += th
        sum_iav_sq += th * th

        mid = mid + 1

    iav = sum_iav / clustersize
    std_iav = math.sqrt((sum_iav_sq - pow(sum_iav, 2) / clustersize) / (clustersize - 1))

    features = [clustersize, std, avg_med_dev, width, linearity, circularity,
                radius, boundary_regularity, mean_curvature, ang_diff, iav, std_iav,
                distance, distance / clustersize]
    return features  #RetuReturn


filename = 'postive_data'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
scanlist = pickle.load(infile)


# print(scanlist.shape)
c = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

featureslist = np.array(c)
featureslist = np.reshape(featureslist,(1,14))


def createFeaturelist():

    for i in range(scanlist.shape[0]):

        scan = scanlist[i]
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
            print('ff',points2.shape)
            points2=np.transpose(points2)

            print('ff',points2.shape)

            W = np.zeros((2,2), np.float64)
            w = np.zeros((2,2), np.float64)
            U = np.zeros((clustersize, 2), np.float64)

            V = np.zeros((2,2), np.float64)

            w,u,vt=cv2.SVDecomp(points2,W,U,V)
            print('ww',w,W)
            rot_points = np.zeros((clustersize,2), np.float64)

            W[0,0]=w[0]
            W[1,1]=w[1]
            rot_points = np.matmul(u,W)
            # print(w)
            # print(u)
            # print(vt)
            # print(rot_points.shape)

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
            features = getFeatures(cluster[:, 0], cluster[:, 1], cluster.shape[0])
            global featureslist
            featureslist = np.append(featureslist, np.array(features).reshape(1,14), axis=0)


createFeaturelist()


filename = 'featurelist'
outfile = open(filename,'wb')
pickle.dump(featureslist,outfile)
print(featureslist.shape)
print("--- %s Time taken ---" % (time.time() - start_time))
