# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: Marcos Yamasaki

"""
import time
start_time = time.time()
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten, TimeDistributed
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import holidays
from sklearn.model_selection import TimeSeriesSplit


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
dataset = dataset[:-140]


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
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegCP','RegSP'], axis=1)


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
# dataset['DATE'] = pd.to_datetime(dataset.Date)

# date = dataset.Date
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'HOUR':dataset.Hour})

# Add weekday to X data
Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

# Add holidays to X data
us_holidays = []
#for date2 in holidays.UnitedStates(years=[2009,2010,2011,2012,2013,2014,2015,2016,2017]).items():
for date2 in holidays.UnitedStates(years=[2011,2012,2013,2014,2015,2016,2017,2018,2019]).items():
    us_holidays.append(str(date2[0]))

Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in us_holidays else 0 for val in date]})



# Concat all new features into X data
concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
# concatlist = [X,Weekday,Holiday]
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
forecastSize = 30*24
testSize = forecastSize/(y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# sc = StandardScaler()
sc = MinMaxScaler()

X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.fit_transform(X_test)



def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def lstmCalc():    
start_time_lstmCalc = time.time()
# global y_test, y_pred, y_train, X_test, X_testsc, X_train, X_trainsc
# X_trainsc_lmse = X_trainsc.reshape(X_trainsc.shape[0],X_trainsc.shape[1],1)
# X_testsc_lmse = X_testsc.reshape(X_testsc.shape[0],X_testsc.shape[1],1)

#y_train_lmse = y_train.values.reshape(y_train.shape[0],1,1)
#y_test_lmse = y_test.values.reshape(y_test.shape[0],1,1)


import nni
from tensorflow.keras.layers import BatchNormalization
params = nni.get_next_parameter()

# EarlyStopping condition
from tensorflow.keras.callbacks import EarlyStopping


_batch = params['batch_size']
_epoch = 150
_neurons = params['neurons_width']
_hidden_layers = params['hidden_layers']
_optimizer = params['optimizer']
_activation = params['activation']

_kernel = 'he_normal'
if params['activation'].find("LeakyReLU_0.01") > -1:  
    _activation = LeakyReLU(alpha=0.01)
elif params['activation'].find("LeakyReLU_0.05") > -1:
    _activation = LeakyReLU(alpha=0.05)
elif params['activation'].find("LeakyReLU_0.1") > -1:
    _activation = LeakyReLU(alpha=0.1)
elif params['activation'] == "selu":
    _kernel = 'lecun_normal'



# n_features = 1

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
r2train_per_fold = []
r2test_per_fold = []
rmse_per_fold = []
mae_per_fold = []
mape_per_fold = []



# convert an array of values into a dataset matrix
#def create_dataset(dataset, look_back=1):
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back-1):
#        a = dataset[i:(i+look_back), 0]
#        dataX.append(a)
#        dataY.append(dataset[i + look_back, 0])
#    return np.array(dataX), np.array(dataY)
#
#look_back = 3
#
#train = np.column_stack((X_train, y_train))
#test = np.column_stack((X_test, y_test))
#X_train, y_train = create_dataset(train, look_back)
#X_test, y_test = create_dataset(test, look_back)


# Merge inputs and targets
#inputs = np.concatenate((X_trainsc_lmse, X_testsc_lmse), axis=0)
inputs = np.concatenate((X_trainsc, X_testsc), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)



# Time Series Split function
# tscv = TimeSeriesSplit(n_splits=5)

kfold = 5
inputs = inputs.reshape(inputs.shape[0],1,inputs.shape[1])
#targets = targets.reshape(targets.shape[0])


# Cross Validation model evaluation fold-5
fold_no = 1
# for train, test in tscv.split(inputs, targets):
#    newInputs = inputs[0][train].reshape(1,inputs[0][train].shape[0],inputs[0][train].shape[1])
#    newTargets = y_train.reshape(1,y_train.shape[0],1)

    # print(len(train))
Ndays=90
test_size = Ndays*24
train_size = round((len(inputs)/kfold) - test_size)

if test_size > train_size:
    print("Test size too high!")

train_index = np.arange(0,train_size)
test_index = np.arange(train_size, train_size+test_size)

for i in range(0, kfold):
    X_train = inputs[train_index]                
    y_train = targets[train_index]
    try:                
        X_test = inputs[test_index]
        y_test = targets[test_index]
    except IndexError:
        test_index = np.arange(test_index[0],len(inputs))
        X_test = inputs[test_index]
        y_test = targets[test_index]



    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # LSTM Implementation
    model = Sequential()
    model.add(LSTM( units=_neurons,
                    activation=_activation,
                    input_shape=[None,X_train.shape[2]],
                    kernel_initializer=_kernel))
    
    if params['dropout'] == "True":
        model.add(Dropout(0.2))
    
    # Adding the hidden layers
    for i in range(_hidden_layers):
        # model.add(LSTM(units=_neurons, activation=_activation, return_sequences=True))
        model.add(Dense(_neurons, activation=_activation, kernel_initializer=_kernel))
    if params['dropout'] == "True":
        model.add(Dropout(0.2))

#    model.add(Flatten())
    # Adding the output layer
    # model.add(TimeDistributed(Dense(1)))
    model.add(Dense(1))
    # Include loss and optimizer functions
    model.compile(loss='mse', optimizer=_optimizer)
    # Save history lstm model and fit the training data
    # Adjust all needed parameters 


    print(model.summary())
    #history_lstm_model = model.fit(X_trainsc_lmse, y_train, epochs=_epoch, batch_size=_batch, verbose=1, shuffle=False, callbacks = [early_stop])

    # history_lstm_model = model.fit(X_trainsc_lmse, y_train,
    early_stop = EarlyStopping(monitor='loss', mode='min', patience=4, verbose=1)
    history_lstm_model = model.fit(X_train, y_train,
                                    epochs=_epoch,        
                                    batch_size=_batch,
                                    verbose=0,
                                    shuffle=False,
                                    callbacks = [early_stop])


#    scores = model.evaluate(X_test, y_test, verbose=0)

    # Predict using test data
    y_pred = model.predict(X_test, batch_size=_batch)
    # Prepare the plot data
    rows = test_index
    # rows = test
    df2 = df.iloc[rows[0]:rows[-1]+1]
#    df2.reset_index(drop=True,inplace=True)
    df = pd.to_datetime(df)
    df2 = pd.to_datetime(df2)

    y_pred = np.float64(y_pred)
    
    # if y_pred.shape[0] == y_test.shape[0]:
    #     y_pred = y_pred.reshape(y_pred.shape[0],1)



    y_pred_train = model.predict(X_train,batch_size=_batch)
    y_pred_train = np.float64(y_pred_train)
    # if y_pred_train.shape[0] == y_train.shape[0]:
    #     y_pred_train = y_pred_train.reshape(y_pred_train.shape[0],1)
    r2train = r2_score(y_train, y_pred_train)
    r2test = r2_score(y_test, y_pred)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2test))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %f" % (mae))

    #mape = mean_absolute_percentage_error(y_test.reshape(y_test.shape[0]), y_pred.reshape(y_pred.shape[0]))
#    mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred.reshape(y_pred.shape[0]))
    mape = mean_absolute_percentage_error(y_test, y_pred.reshape(y_pred.shape[0]))
    
    print("MAPE: %.2f%%" % (mape))


    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores}')
#    acc_per_fold.append(scores * 100)
    loss_per_fold.append(scores)
    r2train_per_fold.append(r2train)
    r2test_per_fold.append(r2test)
    rmse_per_fold.append(rmse)
    mae_per_fold.append(mae)
    mape_per_fold.append(mape)

# Some analysis over predictions made
#    aux_test = pd.DataFrame()    
#    y_pred = np.float64(y_pred)
#    y_pred = y_pred.reshape(y_pred.shape[0])
#    y_test = y_test.reshape(y_test.shape[0])
#    aux_test['error'] = y_test - y_pred
#    aux_test['abs_error'] = aux_test['error'].apply(np.abs)
#    aux_test['DEMAND'] = y_test
#    aux_test['PRED'] = y_pred
#    aux_test['Year'] = X.iloc[test,2].reset_index(drop=True)
#    aux_test['Month'] = X.iloc[test,3].reset_index(drop=True)
#    aux_test['Day'] = X.iloc[test,4].reset_index(drop=True)
#    aux_test['Weekday'] = date.iloc[X.iloc[test,5].shape[0]:].dt.day_name().reset_index(drop=True)
#    aux_test['HOUR'] = X.iloc[test,6].reset_index(drop=True)
#    aux_test['Holiday'] = X.iloc[test,7].reset_index(drop=True)
#
#    error_by_day = aux_test.groupby(['Year','Month','Day','Weekday', 'Holiday']) \
#    .mean()[['DEMAND','PRED','error','abs_error']]
#    print("\nOver forecasted days")
#    print(error_by_day.sort_values(['error'], ascending=[False]).head(10))
#
#    print("\nWorst absolute predicted days")
#    print(error_by_day.sort_values('abs_error', ascending=False).head(10))
#
#    print("\nBest predicted days")
#    print(error_by_day.sort_values('abs_error', ascending=True).head(10))
#
#    error_by_month = aux_test.groupby(['Year','Month']) \
#    .mean()[['DEMAND','PRED','error','abs_error']]
#
#    print("\nOver forecasted months")
#    print(error_by_month.sort_values(['error'], ascending=[False]).head(10))
#
#    print("\nWorst absolute predicted months")
#    print(error_by_month.sort_values('abs_error', ascending=False).head(10))
#
#    print("\nBest predicted months")
#    print(error_by_month.sort_values('abs_error', ascending=True).head(10))

    # Increase fold number
    fold_no = fold_no + 1

    # Increase indexes
    train_index = np.concatenate((train_index, test_index, (train_index + train_size + test_size)), axis=0)
    test_index = test_index + train_size + test_size

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]:.5f}')
    print(f'> Fold {i+1} - r2_score_train: {r2train_per_fold[i]:.5f}')
    print(f'> Fold {i+1} - r2_score_test: {r2test_per_fold[i]:.5f}')
    print(f'> Fold {i+1} - rmse: {rmse_per_fold[i]:.5f}')
    print(f'> Fold {i+1} - mae: {mae_per_fold[i]:.5f}')
    print(f'> Fold {i+1} - mape: {mape_per_fold[i]:.5f}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold):.5f}')
print(f'> r2_score_train: {np.mean(r2train_per_fold):.5f} (+- {np.std(r2train_per_fold):.5f})')
print(f'> r2_score_test: {np.mean(r2test_per_fold):.5f} (+- {np.std(r2test_per_fold):.5f})')
print(f'> rmse: {np.mean(rmse_per_fold):.5f} (+- {np.std(rmse_per_fold):.5f})')
print(f'> mae: {np.mean(mae_per_fold):.5f} (+- {np.std(mae_per_fold):.5f})')
print(f'> mape: {np.mean(mape_per_fold):.5f} (+- {np.std(mape_per_fold):.5f})')
print('------------------------------------------------------------------------')



print("\n--- \t{:0.3f} seconds --- LSTM".format(time.time() - start_time_lstmCalc))
print("\nLSTM has been executed.")


print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))

r2scoreAvg = np.mean(r2test_per_fold)
if r2scoreAvg > 0:
    nni.report_final_result(r2scoreAvg)
else:
    nni.report_final_result(0)
