# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 06:15:54 2020

@author: marko
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:02:14 2019

@author: z003t8hn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import numpy as np
import pandas as pd

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask

#try: 
#    import modin.pandas as pd
#except IOError:
#    print('An error occured trying to read the file.')
#    
#except ValueError:
#    print('Non-numeric data found in the file.')
#
#except ImportError:
#    print ("NO module found")
#    
#except EOFError:
#    print('Why did you do an EOF on me?')
#
#except KeyboardInterrupt:
#    print('You cancelled the operation.')
#
#except:
#    print('An error occured.')
    

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML/ML-Load-Forecasting'
#dataset19 = pd.read_csv(path + r'/datasets/2019_smd_hourly_ISONE CA.csv')
#dataset18 = pd.read_csv(path + r'/datasets/2018_smd_hourly_ISONE CA.csv')
#dataset17 = pd.read_csv(path + r'/datasets/2017_smd_hourly_ISONE CA.csv')
dataset16 = pd.read_csv(path + r'/datasets/2016_smd_hourly_ISONE CA.csv')
dataset15 = pd.read_csv(path + r'/datasets/2015_smd_hourly_ISONE CA.csv')
dataset14 = pd.read_csv(path + r'/datasets/2014_smd_hourly_ISONE CA.csv')
dataset13 = pd.read_csv(path + r'/datasets/2013_smd_hourly_ISONE CA.csv')
#dataset12 = pd.read_csv(path + r'/datasets/2012_smd_hourly_ISONE CA.csv')
#dataset11 = pd.read_csv(path + r'/datasets/2011_smd_hourly_ISONE CA.csv')
#dataset10 = pd.read_csv(path + r'/datasets/2010_smd_hourly_ISONE CA.csv')
#dataset09 = pd.read_csv(path + r'/datasets/2009_smd_hourly_ISONE CA.csv')


#dataset17 = pd.read_csv(r'C:/Users/z003t8hn/code/ML/ML-Load-Forecasting/datasets/2017_smd_hourly_ISONE CA.csv')


#concatlist = [dataset09,dataset10,dataset11,dataset12,dataset13,dataset14,dataset15,dataset16,dataset17]
concatlist = [dataset13,dataset14,dataset15,dataset16]
dataset = pd.concat(concatlist,axis=0,sort=False,ignore_index=True)

## Pre-processing input data 
# Verify zero values in dataset (X,y)
print("Any null value in dataset?")
display(dataset.isnull().any())
print("How many are they?")
display(dataset.isnull().sum())
print("How many zero values?")
display(dataset.eq(0).sum())
print("How many zero values in y (DEMAND)?")
display(dataset['DEMAND'].eq(0).sum())


# Drop unnecessary columns in X dataframe (features)
X = dataset.iloc[:, :]
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP','DryBulb','DewPnt'], axis=1)
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','RT_LMP','RT_EC','RT_CC','RT_MLC'], axis=1)


# Taking care of missing data
if (dataset['DEMAND'].eq(0).sum() > 0):    
    # Replace zero values by NaN
    dataset['DEMAND'].replace(0, np.nan, inplace= True)
    # Save y column (output)
    y = dataset.iloc[:, 3]
    # Replace NaN values by meaningful values
    from sklearn.preprocessing import Imputer
    y_matrix = y.as_matrix()
    y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer = imputer.fit(y_matrix)
    y =  imputer.transform(y_matrix)



date = pd.DataFrame() 
date = pd.to_datetime(dataset.Date)
date.dt.year.head() 
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'Hour':dataset.Hour})

concatlist = [X,Year,Month,Day,Hour]
X = pd.concat(concatlist,axis=1)

test = pd.to_datetime(dataset.Date)
i = 0
i2 = 0
for row in test:
    test[i] = test[i] + pd.DateOffset(hours=1+i2)  
#     print(test[i])
    if (i2 == 23):
         i2 = 0
    else:
        i2 = i2 + 1
    i = i + 1
print(test.head())
df = pd.DataFrame(test)
#concatlist = [X,df]
#X = pd.concat(concatlist,axis=1)



# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)



plt.figure(1)
plt.plot(df,y, color = 'gray', label = 'Real data')
plt.legend()
plt.show()


#from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
#import statsmodels.api as sm
data = pd.DataFrame(data=df)
concatlist = [data,pd.DataFrame(y)]
data = pd.concat(concatlist,axis=1)

data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
result = seasonal_decompose(data[0], model='multiplicative')
#result = sm.tsa.seasonal_decompose(data)
result.plot()
plt.show


#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(3)
rolling_mean = data[0].rolling(window = 12).mean()
rolling_std = data[0].rolling(window = 12).std()
plt.plot(data[0], color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()


result2 = adfuller(data[0])
print('ADF Statistic: {}'.format(result2[0]))
print('p-value: {}'.format(result2[1]))
print('Critical Values:')
for key, value in result2[4].items():
    print('\t{}: {}'.format(key, value))

pvalue = result2[1]


