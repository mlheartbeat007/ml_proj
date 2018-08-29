import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
#create a data frame
df=pd.read_csv('ZILLOW-Z77006_ZRIMFRR.csv')
df.set_index('Date',inplace=True)
#print df.head()
df.to_csv('ihope.csv')
df.plot()
plt.grid()
df.columns=['Houses_price']
print df.head()
plt.show()

