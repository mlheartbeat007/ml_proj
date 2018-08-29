#importing necessary packages
from pandas.io.data import DataReader
from datetime import datetime
from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#downloading data from yahoo
ibm = DataReader('IBM',  'yahoo', datetime(2000,1,1), datetime(2012,1,1))
ibm.info()
ibm.dtypes
ibm.describe()
list(ibm)
ibm.shape


df=ibm['Close']

#arithematic return
ret = df.pct_change()

#geomatric return
#gret = np.log(1 + ret)

ret.head()

#lagged return features
X=pd.concat([ret.shift(1),ret.shift(2),ret.shift(3),ret.shift(4)],axis=1)
X.head()

#tomorrows return
Y=ret.shift(-1)
Y.head()


#combining featrues and target
col=pd.concat([X,Y],axis=1)
col.head()

#removing NaN values
col=col.dropna()

#creating separte features and target variable
feat=col.iloc[:, [i for i in range(col.shape[1]) if i != 4]]
target=col.iloc[:,4]

#training features and target for machine learnging
trainfeat=feat[1:2000]
traintrgt=target[1:2000]

#testing features and target for machine learnging
testfeat=feat[2001:]
testtrgt=target[2001:]

#Support vector regression
clf = svm.SVR()

#training
clf.fit(trainfeat, traintrgt)  

#prediction
prd=clf.predict(testfeat)

#accuracy_score
#testnp = testtrgt.as_matrix()
#accuracy_score(testnp,prd)

