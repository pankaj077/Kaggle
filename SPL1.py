# Packages 
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as auc

#set path
path = 'C:\\Users\\Z013SY0\\Documents\\Python Scripts'
os.chdir(path)
os.getcwd()

#get data 
data_filename= 'train.csv'
data = pd.read_csv(data_filename).set_index("ID")
test_filename= 'test.csv'
test = pd.read_csv(test_filename).set_index("ID")

# remove constants
nunique = pd.Series([data[col].nunique() for col in data.columns], index = data.columns)
constants = nunique[nunique<2].index.tolist()
data = data.drop(constants,axis=1)
test = test.drop(constants,axis=1)

# encode string
strings = data.dtypes == 'object'; strings = strings[strings].index.tolist(); encoders = {}
for col in strings:
    encoders[col] = preprocessing.LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])
    try:
        #why is this transform and not fit_transform
        test[col] = encoders[col].transform(test[col])
    except:
        # lazy way to incorporate the feature only if can be encoded in the test set
        del test[col]
        del data[col]
        
# Get X and y from training set
# Missing values have been replaced with 0s
X = data.drop('target',1).fillna(0); y = data.target        

n_range = [range(10,1000,100)]
n_score=[]
for n in n_range:
    rf=RandomForestClassifier(n_estimators=n)
    score = cross_val_score(rf,X,y,cv=10,scoring="roc_auc").mean() 
    n_score.append(score)
n_optimal = n_range[np.argmax(n_score)]
accuracy_rf = max(n_score)

rf1=RandomForestClassifier(n_estimators=n_optimal) 
rf1.fit(X,y)

from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,530,20)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(rf,X,y,cv=10,scoring="roc_auc") 
    k_score.append(score.mean())
k_optimal = k_range[np.argmax(k_score)]
    
knn1=KNeighborsClassifier(n_neighbors=k_optimal) 
knn1.fit(X,y) 
accuracy_knn = max(k_score)

submission = pd.DataFrame(rf1.fit(X,y).predict_proba(test.fillna(0))[:,1], index=test.index, columns=['target'])






