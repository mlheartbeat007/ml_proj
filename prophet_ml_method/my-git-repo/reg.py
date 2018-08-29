import pandas as pd     #pandas for mathmatical calculation
import quandl			#getting the data 
import unicodecsv
import numpy as np
from tpot import TPOTClassifier,TPOTRegressor
import math 
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
from sklearn import neighbors





#def get_data_from_online():
#data = quandl.get('WIKI/GOOGL')
#data.to_csv('data2.csv')     #get the data
#print df.head()
#  now we have to save the data
#def save_data():
	

data = pd.read_csv('data2.csv')
#print data

#get_the_data_from_csv()

#now we suffle the data
#suffle_data = data.iloc[np.random.permutation(len(data))]
#dat = suffle_data.reset_index(drop=True)
#print dat

#we need specfic in for mation from the data which we needed
df1=data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#print df1.head()

#we need adj.close price volatility pct change adjopen and adj volume
forcast_col = "Adj. Close"
df1.fillna(-999999,inplace=True)
#we gonna predict 1% of the data
forcast_out = int(math.ceil(.01*len(df1)))
#print forcast_out

df1['label'] = df1[forcast_col].shift(-forcast_out)
print df1['label'].head()
X = np.array(df1.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forcast_out]   #data which is known
X_lately = X[-forcast_out:]      #data which is not known
df1.dropna(inplace = True)
Y = np.array(df1['label'])





xtrain,xtest,ytrain,ytest = cross_validation.train_test_split(X,Y,test_size=0.2)



#tpot = TPOTClassifier(generations=5,verbosity=2)
#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)
#tpot.fit(xtrain,ytrain)
#print tpot.score(xtest,ytest)
#tpot.export('tpot_exported_pipeline.py')

clf = neighbors.KNeighborsClassifier()
clf.fit(xtrain,ytrain)
accuracy = clf.score(xtest,ytest)
print accuracy