#---------------------------------------------------------------
## first import the packages for calculating 
##data
import numpy as np
from sklearn import preprocessing ##its for process the data
from sklearn import cross_validation #its for training and testind data
from sklearn import neighbors ##this is the algorithm we use
import pandas as pd   #for working with the csv data
#---------------------------------------------------------------



####### create a dataframe#####
df=pd.read_csv('breast-cancer-wisconsin.csv') 

#-----------------------------------------------------------
## now in this csv file there is a few data that is missing
## and we cant deal with the absense of data
##so what we have to do with replace the data with
## a very big negative data 
#-----------------------------------------------------------

df.replace('?',-99999,inplace=True)

#lets print the datraframe
print df.head()

#----------------------------------------------------------------
##now if you keep the data in the datasets which has no 
## connection with the prediction it will reduce the accuracy
## and creae a serious problem so we need to remove it first
## in this case the id of the patient is the useless column
## because there is no relation between cancer and the patient id
##so drp the id and create a new dataframe
#----------------------------------------------------------------

df.drop(['id'],1,inplace=True)
print df.head()

#--------------------------------------------------------------
## now there is another thing we can use every column except
## the last class column because that is not a feture 
## we gonna predict this information
#------------------------------------------------------------

#----------------------------------------------------------
#in order for calculation we need array of data not 
## the data frame so we need array and only it is possible  
## is with numpy
#----------------------------------------------------------

X = np.array(df.drop(['class'],1))
## this is our feture every thing without the classs
y = np.array(df['class'])
## this is our label that we predict





## now we cross validation the data to find the accuracy
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)





## now its time to use the algorithm
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)



##------------------------------------------------------

#remember the clf.fit() is for training data
#clf.score() for testing data
## basically al this thing do the same 
## after that we can calculate how accurate the data
##------------------------------------------------------
accuracy=clf.score(X_test,y_test)
#print accuracy
## print it if you want to know the accuracy of the program that you using


## lets craeate a user interface that the people can
## input the data

#data =np.array([4,2,1,1,1,2,3,2,1])
#before feed the data we need to reshape the data
#data = data.reshape(1,-1)
#predict=clf.predict(data)
#print predict


def inputfor1():
	clt = int(raw_input('enter the clump thickness \n==>'))
	ucs = int(raw_input('enter the uniform cell size \n==>'))
	ucc = int(raw_input('enter the uniform cell shape \n==>'))
	mad = int(raw_input('enter the marginal_adhession \n==>'))
	secs = int(raw_input('enter the Single Epithelial Cell Size \n==>'))
	bn = int(raw_input('enter the Bare Nuclei  \n==>'))
	bc = int(raw_input('enter the bland cromosome  \n==>'))
	nn = int(raw_input('enter the Normal Nucleoli  \n==>'))
	m = int(raw_input('enter the Mitoses  \n==>'))
	data = np.array([clt,ucs,ucc,mad,secs,bn,bc,nn,m])
	data = data.reshape(1,-1)
	predict = clf.predict(data)
	if predict==2:
		print 'your stage is bengin and the accuracy is '+str(accuracy)
	elif predict==4:
		print 'your stage is malignant and accuracy is '+str(accuracy)
	
def inputformore(ans,listofcancer):
	for item in range(ans):
		clt = int(raw_input('enter the clump thickness \n==>'))
		ucs = int(raw_input('enter the uniform cell size \n==>'))
		ucc = int(raw_input('enter the uniform cell shape \n==>'))
		mad = int(raw_input('enter the marginal_adhession \n==>'))
		secs = int(raw_input('enter the Single Epithelial Cell Size \n==>'))
		bn = int(raw_input('enter the Bare Nuclei  \n==>'))
		bc = int(raw_input('enter the bland cromosome  \n==>'))
		nn = int(raw_input('enter the Normal Nucleoli  \n==>'))
		m = int(raw_input('enter the Mitoses  \n==>'))
		listofcancer.append([clt,ucs,ucc,mad,secs,bn,bc,nn,m])
		status = raw_input('exit?? (yes or no)\n==>')
		if status=='yes':
			break;
	#print len(listofcancer)
	data = np.array(listofcancer)
	data = data.reshape(len(listofcancer),-1)
	predict = clf.predict(data)
	i=1
	for item in predict:

		if item==2:
			print str(i) + " stage is bengien accuracy is "+ str(accuracy)
		else:
			print str(i) + " stage is malignant accuracy is "+ str(accuracy)
		i+=1

def main():
	print "how many patient do you want to process"
	ans = int(raw_input('\n==>'))
	if ans==1:
		inputfor1()
	elif ans>1:
		listofcancer=[]
		inputformore(ans,listofcancer)



main()
