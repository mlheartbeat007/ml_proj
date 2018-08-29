### we create the datafrme as we need first we fetch the data rom the quandl then we edit the whole data frame
##fetching data from the 
## quandl is a website from where we can collect data
import quandl
import pandas as pd

##create a dataframe fetching from the quandl
##quandl.get('FMAC/HPI_TX')
#df.to_csv('four_data.csv')  # for the offline use 

df1=pd.read_csv('four_data.csv')
## to read any csv data with the pandas command is df1=pd.read_csv()




#so we can featch the datafrom the computer
## we can dump it as a pickle but csv is fine
## actually in here you can create a dataframe without 
## the help of pandas 

#print df1.head()









#########################################
#########################################
#########################################

## we can featch the data with the help of the pandas also

## to read fron csv file the command is pd.read_csv()
## to read from the html --> pd.read_html()

df2=pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
#df2.to_html('four_data.html')
#df3=pd.read_html('four_data.html')
## it will display available chart or table as a list not the dictonary
#print df2[0][1:]

##we can create data frame with that list of data
# lets create some list

#data1=[]
#data2=[]
#data3=[]
#data4=[]

#for item in df2:
#	data1.append(df2[0][1])
#	data2.append(df2[0][2])
#	data3.append(df2[0][3])
#	data4.append(df2[0][4])


## now create a datadframe
#print data1[1][1:]
#df3=pd.DataFrame(data1,data2,data3,data4)

#print df
#for abbv in df2[0][1][1:]:
#    print(abbv)


## we take the first item of the list and the second label and all its value

##now we create a ticker that means adding a key value

for abbv in df2[0][1][1:] :
	print 'FMAC/HPI_'+str(abbv)
