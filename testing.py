import numpy as np
import pickle
from sklearn.cluster import DBSCAN
import math
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

filename = 'featurelist'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
featurelist = pickle.load(infile)


print(featurelist.shape)


print(featurelist[0])


data = pd.DataFrame({
                        'feature 1':featurelist[:,0],
                        'feature 2':featurelist[:,1],
                        'feature 3':featurelist[:,2],
                        'feature 4':featurelist[:,3],
                        'feature 5':featurelist[:,4],
                        'feature 6':featurelist[:,5],
                        'feature 7': featurelist[:,6],
                        'feature 8': featurelist[:,7],
                        'feature 9': featurelist[:,8],
                        'feature 10': featurelist[:,9],
                        'feature 11': featurelist[:,10],
                        'feature 12': featurelist[:,11],
                        'feature 13': featurelist[:,12],
                        'feature 14': featurelist[:,13],

})


print(data.head())

data['human'] = 1


X=data[['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6','feature 7', 'feature 8', 'feature 9',
        'feature 10','feature 11','feature 12','feature 13','feature 14']]  # Features



y=data['human']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X.head())


clf=RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')


# clf.fit(X_train,y_train)
clf.fit(X,y)



print(clf.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]))

