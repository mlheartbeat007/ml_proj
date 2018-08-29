#importing 
#-----------------------------------------------------------
from tpot import TPOTClassifier 
#this is for find the best algorithm to calculate
from sklearn import cross_validation
#for training and testing data
import pandas as pd
#work with csv file
import numpy as np
#mathmatical calculation
#-----------------------------------------------------------

### read the csv file 
telescope = pd.read_csv('gamma.csv')

#print telescope.head()
#-------------------------------------------------
#now we suffle the data for a better result
#this is not always necessary
#------------------------------------------------

telescope_suffele = telescope.iloc[np.random.permutation(len(telescope))]
## this iloc function find the index of the value and we randomly assign the value
##then we reset index 
#print telescope_suffele.head()
tele = telescope_suffele.reset_index(drop=True)
### after that the index will again start from 1 but the value will be suffeled

#print tele.head()


##now we need the label which is g for gamma and h for hadron 
## but we wont work with character so we will assign gamma or g for 0
## and h or hadron for 1
## we do it with the map method

##so store the class

tele['class']=tele['class'].map({'g':0,'h':1})
##print tele['class']
##this is my y axix
##now it will also show the id number so we only take the values 
tele_class=tele['class'].values






###### now its time to split the train and testing data
training_part,testing_part = cross_validation.train_test_split(tele.index,stratify=tele_class,train_size=.75,test_size=.25)
##print training_part

## in other program we do it later but now we do it before finding x and y but its
##just splitn the two data one for train and one for test nothing more 

tpot = TPOTClassifier(generations=5,verbosity=2)
X_train= tele.drop('class',axis=1).loc[training_part].values
y_train=tele.loc[training_part,'class'].values

tpot.fit(X_train,y_train)
X_test=tele.drop('class',axis=1).loc[testing_part].values
y_test=tele.loc[testing_part,'class'].values
tpot.score(X_test,y_test)


##now export the genareted code from which we get the algo
tpot.export('pipeline.py')