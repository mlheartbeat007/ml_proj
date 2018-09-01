
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
import time
from sklearn import svm      ## support vector machine packages
############################

############# result conversion
def string_transform(data):
    if data==2:
        value= "benign"
    if data==4:
        value="malignant"
    return value



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



class classifier:

    def svm(self):
            svm_clf=svm.SVC()
            svm_clf.fit(X_train,y_train)
            svm_accuracy=svm_clf.score(X_test,y_test)
            return svm_clf,svm_accuracy
    
    
    def knn(self):
            knn_clf=neighbors.KNeighborsClassifier()
            knn_clf.fit(X_train,y_train)
            knn_accuracy=knn_clf.score(X_test,y_test)
            return knn_clf,knn_accuracy
    
    def gb(self):
            gb_clf = GradientBoostingClassifier()
            #training the data with knn algorithm
            gb_clf.fit(X_train,y_train)

            ## find the knn_accuracy based on the test result
            gb_accuracy=gb_clf.score(X_test,y_test)
            return gb_clf,gb_accuracy
    def rf(self):
        rfc_clf = RandomForestClassifier()
        #training the data with knn algorithm
        rfc_clf.fit(X_train,y_train)

        ## find the knn_accuracy based on the test result
        rfc_accuracy=rfc_clf.score(X_test,y_test)
        return rfc_clf,rfc_accuracy




######## extract the classifier and the accuracy of the classifier

##create an object

clf=classifier()
knn_clf,knn_accuracy=clf.knn()
svm_clf,svm_accuracy=clf.svm()
gb_clf,gb_accuracy=clf.gb()
rf_clf,rf_accuracy=clf.rf()

def print_data():
    time.sleep(.8)
    print "PROCCESSING DATA......"
    time.sleep(.8)
    print "KNN Classifer accuracy        "+str(knn_accuracy)
    time.sleep(.8)
    print "SVM Classifier accuracy        "+str(svm_accuracy)
    time.sleep(.8)
    print "GradientBoosting classifier accuracy "+str(gb_accuracy)
    time.sleep(.8)
    print "Random forest classifier accuracy "+str(rf_accuracy)
    time.sleep(.8)
print_data()


def decision(knn_accuracy,svm_accuracy,gb_accuracy,rfc_accuracy):
    data=[knn_accuracy,svm_accuracy,gb_accuracy,rf_accuracy]
    result=max(data)
    if result == knn_accuracy:
        method = "KNN Classifier"
        working_clf = knn_clf

    elif result==svm_accuracy:
        method='SVC Classifier'
        working_clf = svm_clf
    elif result==gb_accuracy:
        method='GB Classifier'
        working_clf = gb_clf

    else:
        method='RFC Classifier'
        working_clf = rfc_clf
    return method,working_clf


method = decision(knn_accuracy,svm_accuracy,gb_accuracy,rf_accuracy)

best_clf=method[1]
print "BEST METHOD FOUND: "+str(method[0])


def test():
    data1=[[7,4,6,4,6,1,4,3,1],[4,1,1,1,2,1,2,1,1],[4,1,1,1,2,1,3,1,1],[10,7,7,6,4,10,4,1,2],[6,1,1,1,2,1,3,1,1],[7,3,2,10,5,10,5,4,4],[10,5,5,3,6,7,7,10,1]]
    data=np.array(data1)  
    data = data.reshape(len(data1),-1)
    predict = best_clf.predict(data)
    for item in range(len(data1)):
        print  str(item+2)+" patient in stage "+string_transform(predict[item]) +" stage"

    
time.sleep(.8)
print "TESTING SOME RANDOM VALUE..\n"
inp=raw_input('DO YOU WAANT TO TEST\n=>')
if inp =="yes":
    test()
else:
    pass

