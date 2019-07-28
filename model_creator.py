import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

filename = 'rfcmodel'
outfile = open(filename, 'wb')






#Importing featurelist_p1,p2,p3,p4,n1
filename = 'models/modelp1'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
p1 = pickle.load(infile)
filename = 'models/modelp2'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
p2 = pickle.load(infile)
filename = 'models/modelp3'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
p3 = pickle.load(infile)
filename = 'models/modelp4'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
p4 = pickle.load(infile)
filename = 'models/modeln1'
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
n1 = pickle.load(infile)

print(p1.shape)
print(p2.shape)
print(p3.shape)
print(p4.shape)
print(n1.shape)

#Function to create dataframe from arrays
def createdf(px):
    featurelist = px
    featuredata = pd.DataFrame({   'feature 1':featurelist[:,0],
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
                        'feature 14': featurelist[:,13],})
    return featuredata#FunctFun


dfp1 = createdf(p1)
dfp2 = createdf(p2)
dfp3 = createdf(p3)
dfp4 = createdf(p4)
dfn1 = createdf(n1)


dfp5 = pd.concat([dfp1, dfp2, dfp3, dfp4]) #Combining all the positve samples


dfp5['human'] = 1 #setting the category '1' as human
dfn1['human'] = 0 #setting the category '0' as human

data = pd.concat([dfn1, dfp5])

#Setting the featueres. We choose use all the 14 columns as features
X=data[['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6','feature 7', 'feature 8', 'feature 9',
        'feature 10','feature 11','feature 12','feature 13','feature 14']]  # Features



y=data['human']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

rfc = RandomForestClassifier(n_estimators=20, random_state=0)
rfc.fit(X_train,y_train)
# pickle.dump(rfc, outfile)
rfc_predict = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_predict))
print(classification_report(y_test,rfc_predict))
print(accuracy_score(y_test, rfc_predict))

