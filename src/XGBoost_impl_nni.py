# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import time
start_time = time.time()
import numpy as np
import pandas as pd

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt
# import calendar
import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV   #Perforing grid search




# Print configs
pd.options.display.max_columns = None
pd.options.display.width=1000




# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
    

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML-Load-Forecasting'

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONE/csv-fixed/*.csv')

# Select datasets 
#selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
#selectDatasets = ["2015","2016","2017","2018","2019"]

# Initialize dataset list
datasetList = []


# Read all csv files and concat them
for filename in all_files:
    if (filename.find("ISONE") != -1):
        for data in selectDatasets:
            if (filename.find(data) != -1):
                df = pd.read_csv(filename,index_col=None, header=0)
                datasetList.append(df)


# Concat
dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)




## Pre-processing input data 
# Verify zero values in dataset (X,y)
print("Any null value in dataset?")
print(dataset.isnull().any())
print("How many are they?")
print(dataset.isnull().sum())
print("How many zero values?")
print(dataset.eq(0).sum())
print("How many zero values in y (DEMAND)?")
print(dataset['DEMAND'].eq(0).sum())


# Drop unnecessary columns in X dataframe (features)
X = dataset.iloc[:, :]
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP','DRYBULB','DEWPNT'], axis=1)
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)


# Drop additional unused columns/features
for columnNames in X.columns:
    if(columnNames.find("5min") != -1):
        X.drop([columnNames], axis=1, inplace=True)

y = dataset.iloc[:, 3]

# Taking care of missing data
if (dataset['DEMAND'].eq(0).sum() > 0
    or dataset['DEMAND'].isnull().any()):    
    print(dataset[dataset['DEMAND'].isnull()])
    # Replace zero values by NaN
    dataset['DEMAND'].replace(0, np.nan, inplace= True)
    # Save the NaN indexes
    nanIndex = dataset[dataset['DEMAND'].isnull()].index.values
    # Convert to float
    y = dataset['DEMAND'].astype(float)
    # Replace by interpolating zero values
    y = y.interpolate(method='linear', axis=0).ffill().bfill()
    print(y.iloc[nanIndex])
    print("Is there any null values now?\n" + str(y.isnull().any()))

    


# Decouple date and time from dataset
# Then concat the decoupled date in different columns in X data
#date = pd.DataFrame() 
date = pd.to_datetime(dataset.Date)
dataset['DATE'] = pd.to_datetime(dataset.Date)

date = dataset.Date
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'HOUR':dataset.Hour})

# Add weekday to X data
Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

# Add holidays to X data
us_holidays = []
for date2 in holidays.UnitedStates(years=list(map(int,selectDatasets))).items():
    us_holidays.append(str(date2[0]))

Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in us_holidays else 0 for val in date]})



# Concat all new features into X data
concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
X = pd.concat(concatlist,axis=1)

# Set df to x axis to be plot
df = dataset['DATE']

# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)


# Splitting the dataset into the Training set and Test set
# Forecast 30 days - Calculate testSize in percentage
#testSize = (30*24)/(y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler()
# X_trainsc = sc.fit_transform(X)

# X_trainsc = sc.fit_transform(X_train)
# X_testsc = sc.transform(X_test)

# Plot actual data
# plt.figure(1)
# plt.plot(df,y, color = 'gray', label = 'Real data')
# plt.legend()
# plt.ion()
# plt.show()


# class BlockingTimeSeriesSplit():
#     def __init__(self, n_splits):
#         self.n_splits = n_splits
    
#     def get_n_splits(self, X, y, groups):
#         return self.n_splits
    
#     def split(self, X, y=None, groups=None):
#         n_samples = len(X)
#         k_fold_size = n_samples // self.n_splits
#         indices = np.arange(n_samples)

#         margin = 0
#         for i in range(self.n_splits):
#             start = i * k_fold_size
#             stop = start + k_fold_size
#             mid = int(0.8 * (stop - start)) + start
#             yield indices[start: mid], indices[mid + margin: stop]

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




import nni

params = nni.get_next_parameter() 
   
start_time_xgboost = time.time()

# XGBoost
import xgboost

model = xgboost.XGBRegressor(
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],                 
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                n_estimators=params['n_estimators'],                                                                    
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                subsample=params['subsample'],
                seed=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# TimeSeries Split
tscv = TimeSeriesSplit(n_splits=5)       
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
with np.printoptions(precision=3, suppress=True):
    print(scores)
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

# Blocking TimeSeries Split
# print("\nBlocking TimeSeries Split")
# btscv = BlockingTimeSeriesSplit(n_splits=5)
# scores = cross_val_score(model, X_train, y_train, cv=btscv, scoring='r2')    
# with np.printoptions(precision=3, suppress=True):
#     print(scores)
# print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))


# r2score = r2_score(y_test, y_pred)
r2score = scores.mean()
if r2score > 0:
    nni.report_final_result(r2score)
else:
    nni.report_final_result(0)


print("\n--- \t{:0.3f} seconds --- XGBoost".format(time.time() - start_time_xgboost))


aux_test = pd.DataFrame()    
y_pred = np.float64(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0])
# y_test = y_test.reshape(y_test.shape[0])
aux_test['error'] = y_test - y_pred
aux_test['abs_error'] = aux_test['error'].apply(np.abs)
aux_test['DEMAND'] = y_test
aux_test['PRED'] = y_pred
aux_test['Year'] = X_test['Year'].reset_index(drop=True)
aux_test['Month'] = X_test['Month'].reset_index(drop=True)
aux_test['Day'] = X_test['Day'].reset_index(drop=True)
aux_test['Weekday'] = date.iloc[X_train.shape[0]:].dt.day_name().reset_index(drop=True)
aux_test['HOUR'] = X_test['HOUR'].reset_index(drop=True)
aux_test['Holiday'] = X_test['Holiday'].reset_index(drop=True)

error_by_day = aux_test.groupby(['Year','Month','Day','Weekday', 'Holiday']) \
.mean()[['DEMAND','PRED','error','abs_error']]
print("\nOver forecasted days")
print(error_by_day.sort_values(['error'], ascending=[False]).head(10))

print("\nWorst absolute predicted days")
print(error_by_day.sort_values('abs_error', ascending=False).head(10))

print("\nBest predicted days")
print(error_by_day.sort_values('abs_error', ascending=True).head(10))
error_by_month = aux_test.groupby(['Year','Month']) \
.mean()[['DEMAND','PRED','error','abs_error']]

print("\nOver forecasted months")
print(error_by_month.sort_values(['error'], ascending=[False]).head(10))

print("\nWorst absolute predicted months")
print(error_by_month.sort_values('abs_error', ascending=False).head(10))

print("\nBest predicted months")
print(error_by_month.sort_values('abs_error', ascending=True).head(10))
    


print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))


#
## the next command is the last line of my script
#plt.ioff()
#plt.show()