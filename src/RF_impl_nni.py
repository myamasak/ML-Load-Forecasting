# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import time
start_time = time.time()
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt

import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

    
# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML-Load-Forecasting'

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONE/csv-fixed/*.csv')

# Select datasets 
selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]

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
from tensorflow import set_random_seed
set_random_seed(42)


# Splitting the dataset into the Training set and Test set
# Forecast 30 days - Calculate testSize in percentage
# testSize = (30*24)/(y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)



def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


import nni
params = nni.get_next_parameter() 


print("Running Random Forest calculation...")
start_time_randForest = time.time()
# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
model = RandomForestRegressor(n_estimators=params['n_estimators'],
                              max_depth=None if params['max_depth']=="None" else params['max_depth'],
                              min_samples_split=params['min_samples_split'],
                              min_samples_leaf=params['min_samples_leaf'],
                              max_features=params['max_features'],
                              max_leaf_nodes=None if params['max_leaf_nodes']=="None" else params['max_leaf_nodes'],
                              min_impurity_decrease=params['min_impurity_decrease'],
                              bootstrap=True if params['bootstrap']=="True" else False,
                              n_jobs=-1,
                              random_state=42,
                              verbose=0
                              ) 


# TimeSeries Split
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
with np.printoptions(precision=4, suppress=True):
    print(scores)
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

# r2score = r2_score(y_test, y_pred)
r2score = scores.mean()
if r2score > 0:
    nni.report_final_result(r2score)
else:
    nni.report_final_result(0)

print("\n--- \t{:0.3f} seconds --- Random Forest".format(time.time() - start_time_randForest))   


print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))
