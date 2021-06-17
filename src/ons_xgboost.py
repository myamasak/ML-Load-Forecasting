import time

from sklearn.base import is_outlier_detector
start_time = time.time()
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import holidays
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, learning_curve, train_test_split
#import logging
# import BlockingTimeSeriesSplit as btss
# import matplotlib.pyplot as plt


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
Hour = pd.DataFrame({'HOUR':date.dt.hour})


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
    #                 xaxis_title='DATE',
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
#                      width=1000,
#                      height=500)

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
    # Edit the layout
    # fig.update_layout(title=columnName+' Demand outliers fixed',
    #                 xaxis_title='DATE',
    #                 yaxis_title='Demand',
    #                 font=dict(size=26),
    #                 yaxis = dict(
    #                     scaleanchor = "x",
    #                     scaleratio = 1),
    #                 xaxis = dict(
    #                     range=(df[0], df[len(df)-1]),
    #                     constrain='domain'),
    #                 legend=dict(
    #                     yanchor="top",
    #                     y=0.99,
    #                     xanchor="left",
    #                     x=0.01)
    #                 )
#                      width=1000,
#                      height=500)

    # fig.show()
    # fig.write_image("ONS_outliers_fixed_"+columnName+".svg")
    
    return y_

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def xgboostCalc(X_, y_):
    print("Running XGBoost calculation...")
    start_time_xgboost = time.time()
        
    # XGBoost
    import xgboost
    import nni
    params = nni.get_next_parameter() 

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
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_, y_, cv=tscv, scoring='r2')
    with np.printoptions(precision=4, suppress=True):
        print(scores)
    print("Loss: {0:.4f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

    # print("Running XGBoost CrossValidation Blocking Time Series Split...")
    # btscv = btss.BlockingTimeSeriesSplit(n_splits=5)
    # scores = cross_val_score(model, X_trainsc, y_train, cv=btscv, scoring='r2')    
    # with np.printoptions(precision=4, suppress=True):
    #     print(scores)
    # print("Loss: {0:.6f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

    # r2score = r2_score(y_test, y_pred)

    r2score = scores.mean()
    if r2score > 0:
        nni.report_final_result(r2score)
    else:
        nni.report_final_result(0)

    print("\n--- \t{:0.3f} seconds --- XGBoost Cross-validation ".format(time.time() - start_time_xgboost)) 
    

################
# MAIN PROGRAM
################

y_ = outlierDetection(y_=y_)
xgboostCalc(X_=X_, y_=y_)
