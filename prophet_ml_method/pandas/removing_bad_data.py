
##problem:
## there is a ultrasonic sensor at the bottom of the small bridge every time a heavy car pass by the bridge hight reduce a small ammount .but for e certain problem suddenly it shows a giant flactuation and show some big value and then become normal again your job is to create a program that can take the huge ammount of data and it can dynamically remove this unconvenient data.

##the data is given bellow
#
#bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}




import pandas as pd               ##this is needed for creating dataframe
import matplotlib.pyplot as plt   ## needed for plotting
from matplotlib import style      ## needed for style

style.use('fivethirtyeight')


#given bridge height
bridge_height={'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df=pd.DataFrame(bridge_height)
print df

##now plot this to see the unconvenient value by graphics
#df['meters'].plot()
#plt.show()
## lets find the standered deviation of the data that
##means the change of the deviation 
## that means data change rate

## we create a new data frame
df['STD']=pd.rolling_std(df['meters'],2)
#here we chech every consiquent two data from realizing 
## every data changig rate
print df
## actually what we get is every data deviation ammount
##now we create a mean deviation of all
## so that we can create an ideal point 
df_std=df.describe()
print df_std
df_std=df_std['meters']['std']
print df_std

## now create a new dataframe 
df1=df[df['STD']<df_std]
print df1

## now plotting the data
df1['meters'].plot()
plt.show()
#now we got our acctual value
