# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: Marcos Yamasaki

"""
import time
start_time = time.time()
import numpy as np
import pandas as pd
#import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#import datetime as dt
#import calendar
import holidays
from sklearn.model_selection import TimeSeriesSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV   #Perforing grid search
#from sklearn.model_selection import learning_curve
#import BlockingTimeSeriesSplit as btss
#from tensorflow.keras.callbacks import EarlyStopping
#import sys



import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("LSTM_impl.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

print("The program has been started...")


# Print configs
pd.options.display.max_columns = None
pd.options.display.width=1000


# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
    

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML-Load-Forecasting'

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONewEngland/csv-fixed/*.csv')

# Select datasets 
#selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
#selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
selectDatasets = ["2011","2012","2013","2014","2015","2016","2017","2018","2019"]

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
# Adjust to batch_size = 24
dataset = dataset[:-48]

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
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP','DryBulb','DewPnt'], axis=1)
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegCP','RegSP'], axis=1)


# Drop additional unused columns/features
for columnNames in X.columns:
    if(columnNames.find("5min") != -1):
        X.drop([columnNames], axis=1, inplace=True)

y = dataset.iloc[:, 3]

# Taking care of missing data
if (dataset['DEMAND'].eq(0).sum() > 0
    or dataset['DEMAND'].isnull().any()):    
    # Replace zero values by NaN
    dataset['DEMAND'].replace(0, np.nan, inplace= True)
    # Save y column (output)
    y = dataset.iloc[:, 3]
    # Replace NaN values by meaningful values
    # from sklearn.preprocessing import Imputer
    from sklearn.impute import SimpleImputer
    y_matrix = y.to_numpy()
    y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
    # imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(y_matrix)
    y =  imputer.transform(y_matrix)
    y = y.reshape(y.shape[0])


# Decouple date and time from dataset
# Then concat the decoupled date in different columns in X data
date = pd.DataFrame() 
date = pd.to_datetime(dataset.Date)
# dataset['Date'] = pd.to_datetime(dataset.Date)

# date = dataset.Date
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'Hour':dataset.Hour})

# Add weekday to X data
Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

# Add holidays to X data
us_holidays = []
#for date2 in holidays.UnitedStates(years=[2009,2010,2011,2012,2013,2014,2015,2016,2017]).items():
for date2 in holidays.UnitedStates(years=[2012,2013,2014,2015,2016,2017,2018]).items():
    us_holidays.append(str(date2[0]))

Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in us_holidays else 0 for val in date]})



# Concat all new features into X data
concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
# concatlist = [X,Weekday,Holiday]
X = pd.concat(concatlist,axis=1)

# Set df to x axis to be plot
df = dataset['Date']


# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)

from tensorflow import set_random_seed
set_random_seed(42)



# Splitting the dataset into the Training set and Test set
# Forecast 30 days - Calculate testSize in percentage
testSize = (30*24)/(y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
#sc = MinMaxScaler()

#X_trainDrop = X_train.drop(['Date'], axis=1)
X_trainsc = sc.fit_transform(X_train)

#data = pd.DataFrame(data=X_train['Date'])
#concatlist = [data,pd.DataFrame(X_trainsc)]
#data = pd.concat(concatlist,axis=1)
#data.reset_index(drop=True,inplace=True)
#data['Date'] = pd.to_datetime(data['Date'])
##data = data.set_index('Date')
#X_trainsc = data

# data.sort_values('DEMAND', ascending=False).head(10)
# Verify if theres is any null data
#data[data.isnull().any(axis=1)]


#X_testDrop = X_test.drop(['Date'], axis=1)
X_testsc = sc.fit_transform(X_test)

#data = pd.DataFrame(data=X_test['Date'])
#data.reset_index(drop=True,inplace=True)
#concatlist = [data,pd.DataFrame(X_testsc)]
#data = pd.concat(concatlist,axis=1)
#data.reset_index(drop=True,inplace=True)
#data['Date'] = pd.to_datetime(data['Date'])
##data = data.set_index('Date')
#X_testsc = data



# Plot actual data
#plt.figure(1)
#plt.plot(df,y, color = 'gray', label = 'Real data')
#plt.legend()
#plt.ion()
#plt.show()
#plt.savefig('Actual_Data.png')



def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def featImportanceCalc():
    print("Running Feature Importance (RF) calculation...")
    
    start_time_featImportance = time.time()
    
    ## Feature importance
    # Import random forest
    #from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor  
    
    # Create decision tree classifer object
    #clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf = RandomForestRegressor(random_state=0, n_jobs=-1)
    
    # Train model
    model = clf.fit(X, y)
    
    # Calculate feature importances
    importances = model.feature_importances_    
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::]    
    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]
    
    # make a plot with the feature importance
    # plt.figure(figsize=(12,14), dpi= 80, facecolor='w', edgecolor='k')
    plt.figure()
    # plt.grid()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], height=0.2, align='center')
    # plt.axvline(x=0.03)
    plt.yticks(range(len(indices)), list(names))
    plt.xlabel('Relative Importance')   
    plt.show()
    
    # Show plot
    plt.show()
    plt.savefig('Feature_Importance_RF.png')
    
    featImportance = pd.concat([pd.DataFrame({'Features':names}),
                  pd.DataFrame({'Relative_Importance':importances[indices]})],
                    axis=1, sort=False)
    
    print(featImportance)
    
    print("\n--- \t{:0.3f} seconds --- Feature Importance".format(time.time() - start_time_featImportance))


# def lstmCalc():    
start_time_lstmCalc = time.time()
# global y_test, y_pred, y_train, X_test, X_testsc, X_train, X_trainsc
X_trainsc_lmse = X_trainsc.reshape(X_trainsc.shape[0],X_trainsc.shape[1],1)
X_testsc_lmse = X_testsc.reshape(X_testsc.shape[0],X_testsc.shape[1],1)

#y_train_lmse = y_train.values.reshape(y_train.shape[0],1,1)
#y_test_lmse = y_test.values.reshape(y_test.shape[0],1,1)

n_input = X_trainsc_lmse.shape[1]
n_batch = 24
n_epoch = 40
n_neurons = 128
n_features = 1
# n_features = 1

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
r2train_per_fold = []
r2test_per_fold = []
rmse_per_fold = []
mae_per_fold = []
mape_per_fold = []
# Merge inputs and targets
inputs = np.concatenate((X_trainsc_lmse, X_testsc_lmse), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Time Series Split function
tscv = TimeSeriesSplit(n_splits=4)


# Cross Validation model evaluation fold-4
fold_no = 1
for train, test in tscv.split(inputs, targets):
    print(len(train))
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # LSTM Implementation
    model = Sequential()
    model.add(LSTM(units=n_neurons,
                    batch_input_shape=(n_batch, X_trainsc_lmse.shape[1], X_trainsc_lmse.shape[2]))
    #                 input_shape=(n_input,n_features))
    #                 activation='relu',
    #                 kernel_initializer='lecun_uniform',
                    # return_sequences=False,
    #                stateful=True   )             
            )
    # Adding the hidden layers
    for i in range(8):
        model.add(Dense(units = 32))
    #    LReLU = LeakyReLU(alpha=0.05)
    #    model.add(LReLU)
        model.add(Activation('relu'))
        
    #    model.add(Dropout(0.1))

    # Adding the output layer
    model.add(Dense(units = 1))
    # Include loss and optimizer functions
    model.compile(loss='mse', optimizer='RMSProp')
    # Save history lstm model and fit the training data
    # Adjust all needed parameters 


    #early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)


    print(model.summary())
    #history_lstm_model = model.fit(X_trainsc_lmse, y_train, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False, callbacks = [early_stop])

    # history_lstm_model = model.fit(X_trainsc_lmse, y_train,
    history_lstm_model = model.fit(inputs[train], targets[train],
                                    epochs=n_epoch,
                                    batch_size=n_batch,
#                                    validation_split=0.1,
                                    verbose=1,
                                    shuffle=False)

#    scores = model.evaluate(inputs[test], targets[test], verbose=0)

    # Predict using test data
    y_pred = model.predict(inputs[test], batch_size=n_batch)
    # Prepare the plot data
#    rows = X_test.index
    rows = test
    df2 = df.iloc[rows[0]:rows[-1]+1]
#    df2.reset_index(drop=True,inplace=True)
    df = pd.to_datetime(df)
    df2 = pd.to_datetime(df2)

    y_pred = np.float64(y_pred)


    # Plot the result
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - LSTM')
    plt.legend()
    # Show and save the plot
    plt.show()
    plt.savefig('LSTM_pred_fold_'+ str(fold_no) +'.png')

    y_pred_train = model.predict(inputs[train],batch_size=n_batch)
    y_pred_train = np.float64(y_pred_train)
    r2train = r2_score(targets[train], y_pred_train)
    r2test = r2_score(targets[test], y_pred)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2test))

    rmse = np.sqrt(mean_squared_error(targets[test], y_pred))
    print("RMSE: %f" % (rmse))

    mae = mean_absolute_error(targets[test], y_pred)
    print("MAE: %f" % (mae))

    #mape = mean_absolute_percentage_error(y_test.reshape(y_test.shape[0]), y_pred.reshape(y_pred.shape[0]))
#    mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred.reshape(y_pred.shape[0]))
    mape = mean_absolute_percentage_error(targets[test], y_pred.reshape(y_pred.shape[0]))
    
    print("MAPE: %.2f%%" % (mape))


    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores}')
#    acc_per_fold.append(scores * 100)
    loss_per_fold.append(scores)
    r2train_per_fold.append(r2train)
    r2test_per_fold.append(r2test)
    rmse_per_fold.append(rmse)
    mae_per_fold.append(mae)
    mape_per_fold.append(mape)

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
    print(f'> Fold {i+1} - r2_score_train: {r2train_per_fold[i]}')
    print(f'> Fold {i+1} - r2_score_test: {r2test_per_fold[i]}')
    print(f'> Fold {i+1} - rmse: {rmse_per_fold[i]}')
    print(f'> Fold {i+1} - mae: {mae_per_fold[i]}')
    print(f'> Fold {i+1} - mape: {mape_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> r2_score_train: {np.mean(r2train_per_fold)} (+- {np.std(r2train_per_fold)})')
print(f'> r2_score_test: {np.mean(r2test_per_fold)} (+- {np.std(r2test_per_fold)})')
print(f'> rmse: {np.mean(rmse_per_fold)} (+- {np.std(rmse_per_fold)})')
print(f'> mae: {np.mean(mae_per_fold)} (+- {np.std(mae_per_fold)})')
print(f'> mape: {np.mean(mape_per_fold)} (+- {np.std(mape_per_fold)})')
print('------------------------------------------------------------------------')



# print("Running LSTM Learning Curve...")
# start_time_lstmLC = time.time()
        
# print("\n--- \t{:0.3f} seconds --- LSTM Learning curve".format(time.time() - start_time_lstmLC))

print("\n--- \t{:0.3f} seconds --- LSTM".format(time.time() - start_time_lstmCalc))
print("\nLSTM has been executed.")


#lstmCalc()

print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))


#sys.stdout.close().

# the next command is the last line of my script
# plt.ioff()
# plt.show()