
### FIRST IMPORT ALL THE THIRD PARTY PACKAGES

#########################
import pandas as pd     		#pandas is for dataframe handaling
import numpy as np      		#numpy is for array calculation
from sklearn import preprocessing #preprocessing is for scaling and process data
from sklearn import cross_validation  #training and testing data crossvalidation is 
                                       #basically applied for known data to find accuracy

from sklearn import neighbors          #this is the KNN packages 
from sklearn.ensemble import GradientBoostingClassifier  ## gradient boosting packages
                                                         #i use it from experience ;
                                                         #some case it works very good
from sklearn.ensemble import RandomForestClassifier      #random forest packages 

from sklearn import svm      ## support vector machine packages
############################

###first import the whole datasets of with pandas

df=pd.read_csv('breast-cancer-wisconsin.csv')

def process_data(df):
	## we cannot work with the null data so we replace the data 
	## with the very large negative data
	df.replace('?',-99999,inplace=True)   ##cant work with the absesce of data
	df.drop(['id'],1,inplace=True)        ##unnecessary column removing it
	return df
process_data(df)

def prepare_for_cross_validation(df):
	## removing the output so that we can calculate the other
	##by converting it to an array
	X=np.array(df.drop(['class'],1))
	## this is our feture data
	y = np.array(df['class'])
	#this is the feture the output
	return X,y

X,y=prepare_for_cross_validation(df)

##to use multiple algorithm we need a testing data sets and training data sets so that we can find 
##the accuracy of and test the best algorithm first we have to split it up
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
## test_size=.2 means we take the 20% data to test .more data needs more time 
## its enough



print X_train
 