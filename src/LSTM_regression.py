# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:37:29 2019

@author: z003t8hn
"""

import numpy as np
import pandas as pd
import os
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Importing the dataset
dataset17 = pd.read_excel(os.getcwd() + '/datasets/2017_smd_hourly.xls', 'ISO NE CA')
dataset16 = pd.read_excel(os.getcwd() + '/datasets/2016_smd_hourly.xls', 'ISO NE CA')
dataset15 = pd.read_excel(os.getcwd() + '/datasets/2015_smd_hourly.xls', 'ISONE CA')
dataset14 = pd.read_excel(os.getcwd() + '/datasets/2014_smd_hourly.xls', 'ISONE CA')
dataset13 = pd.read_excel(os.getcwd() + '/datasets/2013_smd_hourly.xls', 'ISONE CA')
#dataset12 = pd.read_excel(os.getcwd() + '/datasets/2012_smd_hourly.xls', 'ISONE CA')
#dataset11 = pd.read_excel(os.getcwd() + '/datasets/2011_smd_hourly.xls', 'ISONE CA')
#dataset10 = pd.read_excel(os.getcwd() + '/datasets/2010_smd_hourly.xls', 'ISONE CA')
#dataset09 = pd.read_excel(os.getcwd() + '/datasets/2009_smd_hourly.xls', 'ISONE CA')

# Concatenate datasets into one dataset
#concatlist = [dataset09,dataset10,dataset11,dataset12,dataset13,dataset14,dataset15,dataset16,dataset17]
concatlist = [dataset13,dataset14,dataset15,dataset16,dataset17]
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
    imputer = Imputer(missing_values="NaN", strategy="median", axis=0)
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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)

# scale train and test data to [-1, 1]
#scaler = MinMaxScaler(feature_range=(-1, 1))
#X_trainsc = scaler.fit_transform(X_train)
#X_testsc = scaler.transform(X_test)


#Usually it's a good practice to apply following formula in order to find out
#the total number of hidden layers needed.
#Nh = Ns/(α∗ (Ni + No))
#where
#
#Ni = number of input neurons.
#No = number of output neurons.
#Ns = number of samples in training data set.
#α = an arbitrary scaling factor usually 2-10.

## Initialising the ANN
#model = Sequential()
#
## Adding the input layer and the first hidden layer
#model.add(Dense(16, activation = 'relu', input_dim = 6))
#
## Adding the second hidden layer
#model.add(Dense(units = 16, activation = 'relu'))
#
## Adding the third hidden layer
#model.add(Dense(units = 16, activation = 'relu'))
#
## Adding the output layer
#model.add(Dense(units = 1))
#
##model.add(Dense(1))
## Compiling the ANN
#model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#
## Fitting the ANN to the Training set
#model.fit(X_trainsc, y_train, batch_size = 10, epochs = 200)



#train_sc_df = pd.DataFrame(y_train, columns=['DEMAND'], index=y_train.index)
#test_sc_df = pd.DataFrame(y_test, columns=['DEMAND'], index=y_test.index)
#
#for s in range(1,2):
#    train_sc_df['X_{}'.format(s)] = train_sc_df['DEMAND'].shift(s)
#    test_sc_df['X_{}'.format(s)] = test_sc_df['DEMAND'].shift(s)
#
#X_train = train_sc_df.dropna().drop('DEMAND', axis=1)
#y_train = train_sc_df.dropna().drop('X_1', axis=1)
#
#X_test = test_sc_df.dropna().drop('DEMAND', axis=1)
#y_test = test_sc_df.dropna().drop('X_1', axis=1)


X_train_lmse = X_trainsc.reshape(X_trainsc.shape[0],X_trainsc.shape[1],1)
X_test_lmse = X_testsc.reshape(X_testsc.shape[0],X_trainsc.shape[1],1)

print('Train shape: ', X_train_lmse.shape)
print('Test shape: ', X_test_lmse.shape)
y_train_df = y_train
y_train_df = y_train_df.reshape(y_train_df.shape[0],1)

lstm_model = Sequential()
#lstm_model.add(LSTM(7, input_shape=(6,X_train_lmse.shape[2]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
lstm_model.add(LSTM(X_trainsc.shape[1], input_shape=(X_trainsc.shape[1],1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))

# Adding the second hidden layer
lstm_model.add(Dense(units = 16, activation = 'relu'))

# Adding the third hidden layer
lstm_model.add(Dense(units = 16, activation = 'relu'))

lstm_model.add(Dense(units = 8, activation = 'relu'))

# Adding the output layer
lstm_model.add(Dense(units = 1))

#lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
#early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
#history_lstm_model = lstm_model.fit(X_train_lmse, y_train_df, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
history_lstm_model = lstm_model.fit(X_train_lmse, y_train_df, epochs=100, batch_size=20, verbose=1, shuffle=False)


y_pred_test_lstm = lstm_model.predict(X_test_lmse)
y_train_pred_lstm = lstm_model.predict(X_train_lmse)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))


rows = X_test.index
df2 = df.iloc[rows[0]:]

y_pred = lstm_model.predict(X_test_lmse)
#y_tested = y_test.reset_index()
#y_tested = y_tested.drop(['index'],axis=1)
plt.figure(1)
#plt.plot(df2,y_tested, color = 'red', label = 'Real data')
plt.plot(df,y, color = 'gray', label = 'Real data')
plt.plot(df2,y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

#history_lstm_model = lstm_model.fit(X_train_lmse, y_train_df, epochs=100, batch_size=20, verbose=1, shuffle=False)





#A = pd.DataFrame(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
#b = pd.DataFrame(np.array([11, 12, 13, 14, 15, 16, 17, 18]))
#A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2, random_state = 0, shuffle = False)

