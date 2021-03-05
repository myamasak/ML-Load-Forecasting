import time

start_time = time.time()
import sys
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import holidays
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')

# Save all files in the folder
filename = glob.glob(path + r'/datasets/ONS/*.csv')
filename = filename[0].replace('\\','/')
dataset = pd.read_csv(filename,index_col=None, header=0, delimiter=";")

# Selection of year
selectDatasets = ["2012","2013","2014","2015","2016","2017"]

# Select only selected data
datasetList = []
for year in selectDatasets:
    datasetList.append(dataset[dataset['Data'].str.find(year) != -1])
    
dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)

# replace comma to dot
dataset['Demanda'] = dataset['Demanda'].str.replace(',','.')

# Select X data
X = dataset.iloc[:, :]
X = X.drop(['Demanda'], axis=1)

## Pre-processing input data 
# Verify zero values in dataset (X,y)
print("Any null value in dataset?")
print(dataset.isnull().any())
print("How many are they?")
print(dataset.isnull().sum())
print("How many zero values?")
print(dataset.eq(0).sum())
print("How many zero values in y (Demanda)?")
print(dataset['Demanda'].eq(0).sum())

# Set y
y = dataset['Demanda'].astype(float)

# Taking care of missing data
if (dataset['Demanda'].eq(0).sum() > 0
    or dataset['Demanda'].isnull().any()):    
    print(dataset[dataset['Demanda'].isnull()])
    # Save the NaN indexes
    nanIndex = dataset[dataset['Demanda'].isnull()].index.values
    # Replace zero values by NaN
    dataset['Demanda'].replace(0, np.nan, inplace=True)

    #convert to float
    y = dataset['Demanda'].astype(float)
    
    y = y.interpolate(method='linear', axis=0).ffill().bfill()
    print(y.iloc[nanIndex])


# Select Y data
y = pd.concat([pd.DataFrame({'Demanda':y}), dataset['Subsistema']], axis=1, sort=False)


# Decouple date and time from dataset
# Then concat the decoupled date in different columns in X data

# Transform to date type
X['Data'] = pd.to_datetime(dataset.Data, format="%d/%m/%Y %H:%M")

date = X['Data']
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'Hour':date.dt.hour})


# Add weekday to X data
Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

# Add holidays to X data
br_holidays = []
for date2 in holidays.Brazil(years=list(map(int,selectDatasets))).items():
    br_holidays.append(str(date2[0]))

Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in br_holidays else 0 for val in date]})

# Testing output of holidays
for s in br_holidays:
    if selectDatasets[0] in str(s):
        print(s)


# Concat all new features into X data
concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
#concatlist = [X,Year,Month,Weekday,Day,Hour]
X = pd.concat(concatlist,axis=1)

# Split X data to different subsystems/regions
Xs = X[X['Subsistema'].str.find("Todos") != -1].reset_index(drop=True)
Xs = Xs.drop(['Subsistema','Data'],axis=1)

# Split y data to different subsystems/regions
yall = y[y['Subsistema'].str.find("Todos") != -1]['Demanda'].reset_index(drop=True)

# y = pd.concat([ys, yn, yne, yse, yall], axis=1)
# y.columns = ['South','North','NorthEast','SouthEast','All Regions']

# Save in Date format
df = X[X['Subsistema'].str.find("Todos") != -1]['Data'].reset_index(drop=True)

# Plot south data only
# plt.figure()
# plt.plot(df,yall)
# plt.title('Demand of all regions')
# plt.tight_layout()
# plt.show()
# plt.savefig('ONS_All_Demand_plot')

# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)

from tensorflow import set_random_seed
set_random_seed(42)

Xs = pd.DataFrame(Xs)
#ys = pd.DataFrame(ys)

# Splitting the dataset into the Training set and Test set
# Forecast Ndays - Calculate testSize in percentage
# Ndays = 120
# testSize = (Ndays*24)/(ys.shape[0])
# #testSize = 0.1
# X_train, X_test, y_train, y_test = train_test_split(Xs, yall, test_size = testSize, random_state = 0, shuffle = False)

# y_ = pd.concat([y_train, y_test])
# X_ = pd.concat([X_train, X_test])
y_ = yall
X_ = Xs

def outlierDetection(y_):
    # global X_train, X_test, y_train, y_test, X_
    
    # import plotly.io as pio
    # import plotly.graph_objects as go
    # # import plotly
    # pio.renderers.default = 'browser'
    # pio.kaleido.scope.default_width = 1200
    # pio.kaleido.scope.default_height = 750
    
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=20)

    y_pred = clf.fit_predict(pd.DataFrame(y_))
    # outliers_train = y_train.loc[y_pred_train == -1]
    
    negativeOutlierFactor = clf.negative_outlier_factor_
    outliers = y_.loc[negativeOutlierFactor < (negativeOutlierFactor.mean() - negativeOutlierFactor.std()-1)]
    
    # outliers.reindex(list(range(outliers.index.min(),outliers.index.max()+1)),fill_value=0)
    

    outliers_reindex = outliers.reindex(list(range(df.index.min(),df.index.max()+1)))

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df,
    #                         y=y_,
    #                         name=columnName,
    #                         showlegend=False,
    #                         mode='lines'))                         
    # fig.add_trace(go.Scatter(x=df,
    #                         y=outliers_reindex,
    #                         name='Outliers',
    #                         mode='markers',
    #                         marker_size=10))
    # Edit the layout
    # fig.update_layout(title=columnName+' Demand outliers',
    #                 xaxis_title='Date',
    #                 yaxis_title='Demand',
    #                 font=dict(size=26),
    #                 yaxis = dict(
    #                         scaleanchor = "x",
    #                         scaleratio = 1),
    #                 xaxis = dict(
    #                     range=(df[0], df[len(df)-1]),
    #                     constrain='domain'),
    #                 legend=dict(
    #                     yanchor="top",
    #                     y=0.99,
    #                     xanchor="left",
    #                     x=0.01)
    #                 )
                    #  width=1000,
                    #  height=500)

    # fig.show()
    # fig.write_image("ONS_outliers_"+columnName+".svg")
    
    # Fix outliers by removing and replacing with interpolation
    y_ = pd.DataFrame(y_).replace([outliers],np.nan)    
    y_ = y_.interpolate(method='linear', axis=0).ffill().bfill()
    
    print('Outliers fixed: ', end='\n')
    print(y_.loc[outliers.index.values], end='\n')
    
    # Transform to numpy arrays    
    y_ = np.array(y_)
    y_ = y_.reshape(y_.shape[0])
    
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df,
    #                         y=y_,
    #                         name=columnName,
    #                         mode='lines'))                         
#    fig.add_trace(go.Scatter(x=df,
#                             y=outliers_reindex,
#                             name='Predicted Outliers',
#                             mode='markers',
#                             marker_size=10))
#     Edit the layout
#     fig.update_layout(title=columnName+' Demand outliers fixed',
#                     xaxis_title='Date',
#                     yaxis_title='Demand',
#                     font=dict(size=26),
#                     yaxis = dict(
#                         scaleanchor = "x",
#                         scaleratio = 1),
#                     xaxis = dict(
#                         range=(df[0], df[len(df)-1]),
#                         constrain='domain'),
#                     legend=dict(
#                         yanchor="top",
#                         y=0.99,
#                         xanchor="left",
#                         x=0.01)
#                     )
#                      width=1000,
#                      height=500)

#     fig.show()
#     fig.write_image("ONS_outliers_fixed_"+columnName+".svg")
    
    return y_

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

   

################
# MAIN PROGRAM
################

import nni
params = nni.get_next_parameter() 

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
r2train_per_fold = []
r2test_per_fold = []
rmse_per_fold = []
mae_per_fold = []
mape_per_fold = []

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

# Fix any outlier on dataset
y_ = outlierDetection(y_=y_)


print("Running ANN calculation...")
start_time_ANN = time.time()


# Change variable name because of lazyness
# inputs = X_
# targets = y_
inputs = np.array(X_)
targets = np.array(y_)

kfold = 5

# Cross Validation model evaluation fold-5
fold_no = 1
# Forecast 90 days
Ndays = 90
test_size = Ndays*24
train_size = round((len(inputs)/kfold) - test_size)

# Protection for dumb test size
if test_size > train_size:
    print("Test size too high!")    
    sys.exit()

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
    # ANN Implementation

    # Initializing the ANN
    model = Sequential()

    # Adding the input layer
    model.add(Dense(units=_neurons,
                    activation=_activation,
                    input_dim=X_train.shape[1],
                    kernel_initializer=_kernel))
        
    if params['dropout'] == "True":
        model.add(Dropout(0.2))
    
    # Adding the hidden layers
    for i in range(_hidden_layers):
        model.add(Dense(_neurons, activation=_activation, kernel_initializer=_kernel))
    if params['dropout'] == "True":
        model.add(Dropout(0.2))
            
    # Output layer
    model.add(Dense(1))

    # Compiling the ANN
    model.compile(optimizer = _optimizer, loss = 'mse')

    early_stop = EarlyStopping(monitor='loss', mode='min', patience=5, verbose=1)
    history_ANN_model = model.fit(X_train, y_train,
                                epochs=_epoch,        
                                batch_size=_batch,
                                verbose=0,
                                shuffle=False,
                                callbacks = [early_stop])

    # Predict using test data
    y_pred = model.predict(X_test, batch_size=_batch)
    # Prepare the plot data
    rows = test_index
    # rows = test
    df2 = df.iloc[rows[0]:rows[-1]+1]
    # df2.reset_index(drop=True,inplace=True)
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

    # mape = mean_absolute_percentage_error(y_test.reshape(y_test.shape[0]), y_pred.reshape(y_pred.shape[0]))
    # mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred.reshape(y_pred.shape[0]))
    mape = mean_absolute_percentage_error(y_test, y_pred.reshape(y_pred.shape[0]))
    
    print("MAPE: %.2f%%" % (mape))


    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores}')
    # acc_per_fold.append(scores * 100)
    loss_per_fold.append(scores)
    r2train_per_fold.append(r2train)
    r2test_per_fold.append(r2test)
    rmse_per_fold.append(rmse)
    mae_per_fold.append(mae)
    mape_per_fold.append(mape)


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


print("\n--- \t{:0.3f} seconds --- ANN".format(time.time() - start_time_ANN))
print("\nANN has been executed.")


print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))

r2scoreAvg = np.mean(r2test_per_fold)
if r2scoreAvg > 0:
    nni.report_final_result(r2scoreAvg)
else:
    nni.report_final_result(0)