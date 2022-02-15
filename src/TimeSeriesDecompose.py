from tensorflow import set_random_seed
import random
import math
from pandas.plotting import register_matplotlib_converters
import nni
from sklearn.preprocessing import MinMaxScaler, normalize
from skgarden import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, VotingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn import svm
from sklearn import linear_model, cross_decomposition
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import ewtpy
from PyEMD import EMD, EEMD, CEEMDAN
import sys
from Results import Results
from scipy import stats, special
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
import holidays
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import glob
import os
import numpy as np
import pandas as pd
import json
# from dm_test.dm_test import dm_test
"""
Time-Series Decomposition
Author: Marcos Yamasaki
04/03/2021
"""
import time
from log import log
import logging
start_time = time.time()
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV
# from vmdpy import VMD
#from RobustSTL import RobustSTL
plt.close("all")
register_matplotlib_converters()
sys.path.append('../')
### Constants ###
# Dataset chosen
# DATASET_NAME = 'isone'
DATASET_NAME = 'ONS'
# Enable nni for AutoML
enable_nni = False
# Set True to plot curves
PLOT = True
SAVE_FIG = True
# Configuration for Forecasting
ALGORITHM = 'xgboost'
CROSSVALIDATION = True
KFOLD = 10
OFFSET = 0
FORECASTDAYS = 15
NMODES = 1
MODE = 'stl-a'
BOXCOX = True
STANDARDSCALER = True
MINMAXSCALER = False
DIFF = False
LOAD_DECOMPOSED = True
RECURSIVE = False
GET_LAGGED = False
PREVIOUS = False
HYPERPARAMETER_TUNING = True
HYPERPARAMETER_IMF = 'Trend'
STEPS_AHEAD = 24*1
TEST_DAYS = 29
MULTIMODEL = True
LSTM_ENABLED = False
FINAL_TEST = True
FINAL_TEST_ONLY = False
FINAL_FOLD = 20
SAVE_JSON = True
LOOP = False
# Selection of year
selectDatasets = ["2015", "2016", "2017", "2018", "2019"]
# selectDatasets = ["2017","2018"]
# Seed Random Numbers with the TensorFlow Backend
SEED_VALUE = 4242
random.seed(SEED_VALUE)
set_random_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
###

# Default render
pio.renderers.default = 'browser'
# Default size for plotly export figures
#pio.kaleido.scope.default_width = 1280
#pio.kaleido.scope.default_height = 720
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (14, 6)})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(
    f'\\{os.path.basename(__file__)}', '').replace('\\', '/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML', '')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src', '')
elif path.find('loop') != -1:
    path = r'%s' % path.replace('/src/loop.py', '')
    
print(f"path = {path}")



 # LSTM parameters
_batch = 24
_epochs = 50
_neurons = 128
_hidden_layers = 4
_optimizer = 'Adam'
_dropout = False
_dropoutVal = 0.2
_activation = LeakyReLU(alpha=0.2)
LSTM_PARAMS = {
                "optimizer":_optimizer,
                "neurons_width":_neurons,
                "hidden_layers":_hidden_layers,
                "activation":_activation,
                "batch_size":_batch,
                "epochs":_epochs,
                "dropout":_dropout,
                "dropout_val":_dropoutVal
                }
# LSTM Implementation
# model, early_stop = init_lstm(LSTM_PARAMS)

def regressors(algorithm : str):
    REGRESSORS = {  "knn": KNeighborsRegressor(),
                    "dt": DecisionTreeRegressor(random_state=SEED_VALUE),
                    "rf": RandomForestRegressor(random_state=SEED_VALUE),
                    "svr": svm.SVR(),
                    "xgboost": xgboost.XGBRegressor(random_state=SEED_VALUE),
                    "gbr": GradientBoostingRegressor(random_state=SEED_VALUE),
                    "extratrees": ExtraTreesRegressor(random_state=SEED_VALUE),
                    "ard": linear_model.ARDRegression(),
                    "sgd": linear_model.SGDRegressor(random_state=SEED_VALUE),
                    "bayes": linear_model.BayesianRidge(),
                    "lasso": linear_model.LassoLars(),
                    "par": linear_model.PassiveAggressiveRegressor(random_state=SEED_VALUE),
                    "theilsen": linear_model.TheilSenRegressor(random_state=SEED_VALUE),
                    "linear": linear_model.LinearRegression()
                 }

    return REGRESSORS[algorithm]


def datasetImport(selectDatasets, dataset_name='ONS'):
    log('Dataset import has been started')
    # Save all files in the folder
    if dataset_name.find('ONS') != -1:
        filename = glob.glob(path + r'/datasets/ONS/*allregions*.csv')
        filename = filename[0].replace('\\', '/')
        dataset = pd.read_csv(filename, index_col=None,
                              header=0, delimiter=";")
        # Select only selected data
        datasetList = []
        for year in selectDatasets:
            datasetList.append(dataset[dataset['DATE'].str.find(year) != -1])
    elif dataset_name.find('isone') != -1:
        all_files = glob.glob(
            path + r'/datasets/ISONE/csv-fixed/*.csv')
        # Initialize dataset list
        datasetList = []
        # Read all csv files and concat them
        for filename in all_files:
            if (filename.find("ISONE") != -1):
                for data in selectDatasets:
                    if (filename.find(data) != -1):
                        df = pd.read_csv(filename, index_col=None, header=0)
                        datasetList.append(df)
    # Concat them all
    dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)

    if dataset_name.find('ONS') != -1:
        # replace comma to dot
        dataset['DEMAND'] = dataset['DEMAND'].str.replace(',', '.')
        dataset['DATE'] = pd.to_datetime(dataset.DATE, format="%d/%m/%Y %H:%M")
        dataset = dataset.sort_values(by='DATE', ascending=True)

    # test_set = dataset.iloc[:-24*60,:]
    # dataset = dataset.iloc[:-24*60,:]
    return dataset


def dataCleaning(dataset, dataset_name='ONS'):
    log('Data cleaning function has been started')
    # Select X data
    X = dataset.iloc[:, :]
    if dataset_name.find('ONS') != -1:
        X = X.drop(['DEMAND'], axis=1)
    elif dataset_name.find('isone') != -1:
        try:
            X = X.drop(['DEMAND', 'DA_DEMD', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC', 'DATE', 'HOUR',
                        'RT_LMP', 'RT_EC', 'RT_CC', 'RT_MLC', 'SYSLoad', 'RegSP', 'RegCP'], axis=1)
        except KeyError:
            X = X.drop(['DEMAND', 'DA_DEMD', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC',
                        'DATE', 'HOUR', 'RT_LMP', 'RT_EC', 'RT_CC', 'RT_MLC', 'SYSLoad'], axis=1)
        # Drop additional unused columns/features
        for columnNames in X.columns:
            if(columnNames.find("5min") != -1):
                X.drop([columnNames], axis=1, inplace=True)
    # Pre-processing input data
    # Verify zero values in dataset (X,y)
    log("Any null value in dataset?")
    log(dataset.isnull().any())
    log("How many are they?")
    log(dataset.isnull().sum())
    log("How many zero values?")
    log(dataset.eq(0).sum())
    log("How many zero values in y (DEMAND)?")
    log(dataset['DEMAND'].eq(0).sum())

    # Set y
    y = dataset['DEMAND'].astype(float)

    # Taking care of missing data
    log('Taking care of missing data')
    if (dataset['DEMAND'].eq(0).sum() > 0
            or dataset['DEMAND'].isnull().any()):
        log(dataset['DEMAND'][dataset['DEMAND'].isnull()])
        # Save the NaN indexes
        nanIndex = dataset[dataset['DEMAND'].isnull()].index.values
        # Replace zero values by NaN
        dataset['DEMAND'].replace(0, np.nan, inplace=True)
        # convert to float
        y = dataset['DEMAND'].astype(float)
        y = y.interpolate(method='linear', axis=0).ffill().bfill()
        log(y.iloc[nanIndex])

    # Select Y data
    if dataset_name.find('ONS') != -1:
        # y = pd.concat([pd.DataFrame({'DEMAND':y}), dataset['SUBSYSTEM']], axis=1, sort=False)
        y = pd.DataFrame({'DEMAND': y})

    return X, y


def featureEngineering(dataset, X, y, selectDatasets, weekday=True, holiday=True, holiday_bridge=False, demand_lag=True, dataset_name='ONS'):
    log('Feature engineering has been started')
    # Decouple date and time from dataset
    # Then concat the decoupled date in different columns in X data

    # It will not work ----
    # log("Use lagged y (demand) to include as input in X")
    # X, y = get_lagged_y(X, y, n_steps=1)

    log("Adding date components (year, month, day, holidays and weekdays) to input data")
    # Transform to date type
    X['DATE'] = pd.to_datetime(dataset.DATE)

    date = X['DATE']
    Year = date.dt.year.rename('Year')
    Month = date.dt.month.rename('Month')
    Day = date.dt.day.rename('Day')
    Hour = date.dt.hour.rename('HOUR')

    if weekday:
        # Add weekday to X data
        Weekday = date.dt.dayofweek.rename('Weekday')

    if holiday:
        # Add holidays to X data
        br_holidays = []
        for date2 in holidays.Brazil(years=list(map(int, selectDatasets))).items():
            br_holidays.append(str(date2[0]))

        # Set 1 or 0 for Holiday, when compared between date and br_holidays
            Holiday = pd.DataFrame(
                {'Holiday': [1 if str(val).split()[0] in br_holidays else 0 for val in date]})

    # Concat all new features into X data
    try:
        concatlist = [X, Year, Month, Day, Weekday, Hour, Holiday]
    except (AttributeError, ValueError, KeyError, UnboundLocalError) as e:
        concatlist = [X, Year, Month, Day, Hour]
    X = pd.concat(concatlist, axis=1)

    # Split X data to different subsystems/regions
    # Xs = X[X['SUBSYSTEM'].str.find("South") != -1].reset_index(drop=True)
    # Xs = Xs.drop(['SUBSYSTEM','DATE'],axis=1)

    # Save in Date format
    global df  # set a global variable for easier plot
    if dataset_name.find('ONS') != -1:
        # df = X[X['SUBSYSTEM'].str.find("All") != -1]['DATE'].reset_index(drop=True)
        df = X['DATE'].reset_index(drop=True)
    elif dataset_name.find('isone') != -1:
        df = X['DATE'].reset_index(drop=True)

    if holiday_bridge:
        log("Adding bridge days (Mondays / Fridays) to the Holiday column")
        # Holidays on Tuesdays and Thursday may have a bridge day (long weekend)
        # X_tmp = X[(X['Holiday'] > 0).values].drop_duplicates(subset=['Day','Month','Year'])
        X_tmp = X[(X['Holiday'] > 0).values]
        # Filter holidays set on Tuesdays and add bridge day on Mondays
        # 0 = Monday; 1 = Tuesday; ...; 6 = Sunday
        # Start with Tuesdays
        X_tuesdays = X_tmp[X_tmp['Weekday'] == 1]
        bridgeDayList = []
        for tuesday in X_tuesdays['DATE']:
            # Go back one day (monday)
            bridge_day = tuesday - pd.DateOffset(days=1)
            bridgeDayList.append(bridge_day)

        # Do the same for Thursday
        X_thursdays = X_tmp[X_tmp['Weekday'] == 3]
        for thursday in X_thursdays['DATE']:
            # Go back one day (Friday)
            bridge_day = thursday + pd.DateOffset(days=1)
            bridgeDayList.append(bridge_day)

        Holiday_bridge = pd.DataFrame(
            {'Holiday_bridge': [1 if val in bridgeDayList else 0 for val in date]})

        concatlist = [X, Holiday_bridge]
        X = pd.concat(concatlist, axis=1)

        # Sum the two holidays columns to merge them into one and remove unnecessary columns
        X['Holiday_&_bridge'] = X.loc[:, [
            'Holiday', 'Holiday_bridge']].sum(axis=1)
        X = X.drop(['Holiday', 'Holiday_bridge'], axis=1)

    if dataset_name.find('ONS') != -1:
        # Store regions in a list of dataframes
        log('Drop SUBSYSTEM column')
        X = X.drop('SUBSYSTEM', axis=1)

    # elif dataset_name.find('isone') != -1:
    #     X.append(X)
    #     y.append(y)

    return X, y


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def maep(y_true, y_pred):
    return (mean_absolute_error(y_true, y_pred) / np.mean(y_true))*100
    

def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def calc_r2score(y_true, y_pred):
    y_true = np.array([y_true])
    y_pred = np.array([y_pred])

    try:
        if y_true.size != y_pred.size:
            raise Exception('y_true length is different than y_pred')
        elif y_true.size == 1 and y_pred.size == 1:
            return 1-abs(y_true-y_pred)/y_true
    except (AttributeError, ValueError, TypeError) as e:
        raise e

    RSS = sum(np.power((y_true - y_pred), 2))
    TSS = sum(np.power((y_true - np.mean(y_true)), 2))
    result = 1 - (RSS/TSS)
    return result


def decomposeSeasonal(X_, y_, dataset_name='ONS', Nmodes=3, mode='stl-a', final_test=False):
    tic = time.time()
    if mode == 'stl-a' or mode == 'stl-m':
        log('Seasonal and Trend decomposition using Loess (STL) Decomposition has been started')
        data = pd.DataFrame(X_)

        if dataset_name.find('ONS') != -1:
            try:
                concatlist = [data, pd.DataFrame(
                    y_.drop(['SUBSYSTEM'], axis=1))]
            except (AttributeError, KeyError) as e:
                concatlist = [data, pd.DataFrame(y_)]
        elif dataset_name.find('isone') != -1:
            concatlist = [data, pd.DataFrame(y_)]
        data = pd.concat(concatlist, axis=1)

        data.reset_index(inplace=True)
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data.set_index('DATE')
        data = data.drop(['index'], axis=1)
        # data.columns = ['DEMAND']
        if mode == 'stl-a':
            model = 'additive'
        elif mode == 'stl-m':
            model = 'multiplicative'
        result = seasonal_decompose(
            data['DEMAND'], period=24, model=model, extrapolate_trend='freq')
        result.trend.reset_index(drop=True, inplace=True)
        result.seasonal.reset_index(drop=True, inplace=True)
        result.resid.reset_index(drop=True, inplace=True)
        result.observed.reset_index(drop=True, inplace=True)
        # result.trend.name = 'Trend'
        # result.seasonal.name = 'Seasonal'
        # result.resid.name = 'Residual'
        # result.observed.name = 'Observed'
        df_trend = pd.DataFrame({'Trend': result.trend})
        df_seasonal = pd.DataFrame({'Seasonal': result.seasonal})
        df_resid = pd.DataFrame({'Residual': result.resid})
        df_observed = pd.DataFrame({'Observed': result.observed})
        decomposeList = [df_trend, df_seasonal, df_resid]

        # Select one component for seasonal decompose
        # REMOVE FOR NOW
        # log(f'Seasonal component choosen: {seasonal_component}')
        # for component in decomposeList:
        #     if (seasonal_component == component.columns[0]):
        #         y = component
        #         break
        toc = time.time()
        log(f"{toc-tic:0.3f} seconds - Seasonal and Trend decomposition using Loess (STL) Decomposition has finished.")
    elif mode == 'emd' or mode == 'eemd' or mode == 'vmd' or mode == 'ceemdan' or mode == 'ewt':
        decomposeList = emd_decompose(
            y_, Nmodes=Nmodes, dataset_name=DATASET_NAME, mode=mode, final_test=final_test)
    elif mode == 'robust-stl':
        labels = ['Observed', 'Trend', 'Seasonal', 'Remainder']
        if LOAD_DECOMPOSED:
            all_files = glob.glob(
                path + r"/datasets/"+ DATASET_NAME + "/custom/robust-stl*.csv")
            # Initialize dataset lis t
            decomposeList = []
            i = 0
            concat = []
            # Read all csv files and concat them
            for filename in all_files:
                if filename.find(MODE) != -1:
                    df = pd.read_csv(filename, index_col=None, header=None)
                    concat.append(df)
                if i >= 3:
                    decomposeList.append(pd.concat([concat[0], concat[1], concat[2], concat[3]], axis=0).drop(
                        'DATE', axis=1).reset_index(drop=True))
                    concat = []
                    i = -1
                i += 1

        else:
            decomposeList = RobustSTL(y_.values.ravel(
            ), 50, reg1=10.0, reg2=0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
            for i in range(len(decomposeList)):
                decomposeList[i] = pd.DataFrame({labels[i]: decomposeList[i]})
        return decomposeList

    elif mode == 'none':
        decomposeList = [y_]

    return decomposeList


def outlierCleaning(y_, columnName='DEMAND', dataset_name='ONS'):
    # global X_train, X_test, y_train, y_test, X_
    # Drop subsystem and date columns
    if dataset_name.find('ONS') != -1:
        try:
            if y_.columns[1].find("SUBSYSTEM") != -1:
                y_ = y_.drop(['SUBSYSTEM'], axis=1)
            else:
                y_ = y_
        except (AttributeError, IndexError) as e:
            y_ = y_

    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=25)

    y_pred = clf.fit_predict(pd.DataFrame(y_))
    # outliers_train = y_train.loc[y_pred_train == -1]

    negativeOutlierFactor = clf.negative_outlier_factor_
    outliers = y_.loc[negativeOutlierFactor < (
        negativeOutlierFactor.mean() - negativeOutlierFactor.std()-1)]

    # outliers.reindex(list(range(outliers.index.min(),outliers.index.max()+1)),fill_value=0)

    outliers_reindex = outliers.reindex(
        list(range(df.index.min(), df.index.max()+1)))
    if PLOT and False:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df,
                                 y=y_.squeeze(),
                                 name=columnName,
                                 showlegend=False,
                                 mode='lines'))
        fig.add_trace(go.Scatter(x=df,
                                 y=outliers_reindex.squeeze(),
                                 name='Outliers',
                                 mode='markers',
                                 marker_size=10))
        # Edit the layout
        fig.update_layout(title=columnName+' Demand outliers',
                          xaxis_title='DATE',
                          yaxis_title='Demand',
                          font=dict(size=26),
                          yaxis=dict(
                              scaleanchor="x",
                              scaleratio=1),
                          xaxis=dict(
                              range=(df[0], df[len(df)-1]),
                              constrain='domain'),
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01)
                          )
    #                      width=1000,
    #                      height=500)

        fig.show()
        if SAVE_FIG:
            fig.write_image(f"{path}{DATASET_NAME}_outliers_"+columnName+".pdf")

    # Fix outliers by removing and replacing with interpolation
    try:
        y_ = y_.replace([outliers], np.nan)
    except (ValueError, KeyError) as e:
        y_ = y_.replace(outliers, np.nan)
    y_ = y_.interpolate(method='linear', axis=0).ffill().bfill()

    print('Outliers fixed: ', end='\n')
    print(y_.loc[outliers.index.values], end='\n')

    # Transform to numpy arrays
    y_ = np.array(y_)
    y_ = y_.reshape(y_.shape[0])

    if PLOT and False:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df,
                                 y=y_,
                                 name=columnName,
                                 mode='lines'))
    #    fig.add_trace(go.Scatter(x=df,
    #                             y=outliers_reindex,
    #                             name='Predicted Outliers',
    #                             mode='markers',
    #                             marker_size=10))
        # Edit the layout
        fig.update_layout(title=columnName+' Demand outliers fixed',
                          xaxis_title='DATE',
                          yaxis_title='Demand',
                          font=dict(size=26),
                          yaxis=dict(
                              scaleanchor="x",
                              scaleratio=1),
                          xaxis=dict(
                              range=(df[0], df[len(df)-1]),
                              constrain='domain'),
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01)
                          )
    #                      width=1000,
    #                      height=500)

        fig.show()
        if SAVE_FIG:
            fig.write_image(f"{DATASET_NAME}_outliers_fixed_"+columnName+".pdf")

    return y_


def loadForecast(X, y, CrossValidation=False, kfold=5, offset=0, forecastDays=15, dataset_name='ONS'):
    log("Load Forecasting algorithm has been started")
    start_time_loadForecast = time.time()

    global df, fig
    kfoldPred = []
    # Plot
    if PLOT:
        # fig = go.Figure()
        plt.figure()

    # Drop subsystem and date columns
    X, y = data_cleaning_columns(X, y)

    # Shift demand and drop null values
    if GET_LAGGED:
        X, y = get_lagged_y(X, y, n_steps=1)
        if len(df) > len(y):
            # df = df[:len(y)]
            df = df.drop(index=0).reset_index(drop=True)

    # Drop unnecessary columns from X
    # if y.columns[0].find('IMF_0') != -1:
    #     X = X.drop(['Year','Month','Day','Weekday','Holiday','HOUR','DRYBULB','DEWPNT'], axis=1)
    # elif y.columns[0].find('IMF_1') != -1:
    #     X = X.drop(['Year','HOUR','Month','DRYBULB','Holiday'], axis=1)
    # elif y.columns[0].find('IMF_2') != -1:
    #     X = X.drop(['Year','Holiday','HOUR','DRYBULB'], axis=1)
    # elif y.columns[0].find('IMF_3') != -1:
    #     X = X.drop(['Year','Month','Day','Holiday','Weekday','DEWPNT','DRYBULB'], axis=1)
    # elif y.columns[0].find('IMF_4') != -1:
    #     X = X.drop(['Year','Month','Day','Holiday','DRYBULB','DEWPNT','Weekday','HOUR'], axis=1)

    # Define test size by converting days to percentage
    # testSize = 0.05
    testSize = forecastDays*24/X.shape[0]

    if CrossValidation:
        log('CrossValidation has been started')
        log(f'Predict {kfold}-folds each by {testSize*X.shape[0]/24} days')
        log(f'Prediction on decomposed part: {y.columns[0]}')

        # Change variable name because of lazyness
        inputs = np.array(X)
        targets = np.array(y)

        # Rest fold number
        fold_no = 1

        # Forecast X days
        # uniqueYears = X['Year'].unique()
        # test_size = round((X.shape[0]/uniqueYears.size)/12/2)
        test_size = round(forecastDays*24)
        train_size = math.floor((len(inputs)/kfold) - test_size)

        # Offset on Forecast window
        # offset = test_size*3

        if offset > 0:
            log(f'Offset has been set by {offset/24} days')
            # test_size = round((X.shape[0]-offset)/uniqueYears.size/12/2)
            test_size = round(forecastDays*24)
            train_size = math.floor(((len(inputs)-offset)/kfold) - test_size)

        train_index = np.arange(0, train_size+offset)
        test_index = np.arange(train_size+offset, train_size+test_size+offset)

        # Add real data to PLOT
        if PLOT:
            # fig.add_trace(go.Scatter(x=df,
            #                             y=y.squeeze(),
            #                             name=f'Electricity Demand [MW] - {y.columns[0]}',
            #                             mode='lines'))
            # # Edit the layout
            # fig.update_layout(title=f'{dataset_name} dataset Load Forecasting - Cross-Validation of {kfold}-fold',
            #                     xaxis_title='DATE',
            #                     yaxis_title=f'Demand Prediction [MW] - {y.columns[0]}'
            #                     )
            if BOXCOX:
                plt.title(
                    f'Electricity Prediction [MW] - with Box-Cox Transformation - {y.columns[0]}')
                plt.ylabel(f'Load [MW] - Box-Cox')
            else:
                plt.title(f'Electricity Prediction [MW] - {y.columns[0]}')
                plt.ylabel(f'Load [MW] - {y.columns[0]}')
            plt.xlabel(f'Date')
            plt.plot(df, y.squeeze(), color='darkgray', label='Real data')

        if not enable_nni:
            # model = xgboost.XGBRegressor()
            #                             colsample_bytree=0.8,
            #                             gamma=0.3,
            #                             learning_rate=0.03,
            #                             max_depth=7,
            #                             min_child_weight=6.0,
            #                             n_estimators=1000,
            #                             reg_alpha=0.75,
            #                             reg_lambda=0.01,
            #                             subsample=0.95,
            #                             seed=42)
            # Best configuration so far: gbr; metalearner=ARDR
            # regressors = list()
            # regressors.append(('xgboost', xgboost.XGBRegressor()))
            # regressors.append(('knn', KNeighborsRegressor()))
            # regressors.append(('cart', DecisionTreeRegressor()))
            # regressors.append(('rf', RandomForestRegressor()))
            # regressors.append(('svm', svm.SVR()))
            # regressors.append(('gbr', GradientBoostingRegressor()))
            # regressors.append(('extratrees', ExtraTreesRegressor()))
            # regressors.append(('sgd', linear_model.SGDRegressor()))
            # regressors.append(('bayes'  , linear_model.BayesianRidge()))
            # regressors.append(('lasso', linear_model.LassoLars()))
            # regressors.append(('ard', linear_model.ARDRegression()))
            # regressors.append(('par', linear_model.PassiveAggressiveRegressor()))
            # regressors.append(('theilsen', linear_model.TheilSenRegressor()))
            # regressors.append(('linear', linear_model.LinearRegression()))

            # define meta learner model
            # meta_learner = linear_model.ARDRegression()  # 0.88415

            # model = VotingRegressor(estimators=regressors)
            # model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)
            # model = StackingRegressor(
            #     estimators=regressors, final_estimator=meta_learner)

            model = regressors(ALGORITHM)

            # Choose one model for each IMF
            if MULTIMODEL and MODE != 'none':
                if y_decomposed.columns[0].find('IMF_') != -1 or \
                   y_decomposed.columns[0].find('Trend') != -1 or \
                   y_decomposed.columns[0].find('Residual') != -1 or \
                   y_decomposed.columns[0].find('Seasonal') != -1:                    
                    model = regressors(ALGORITHM)
                    local_params = open_json(model, ALGORITHM, y.columns[0])
            else: # for individual algorithm Manual tuning                
                local_params = open_json(model, ALGORITHM, 'none', manual=True)
            # Set model hyperparameters from json file
            model.set_params(**local_params)
        else:  # nni enabled
            
            # Convert some params to int to avoid error (float bug)
            if ALGORITHM == 'gbr' or \
               ALGORITHM == 'extratrees' or \
               ALGORITHM == 'rf':
                if params['min_samples_split'] > 1:
                    params['min_samples_split'] = int(params['min_samples_split'])
                else:
                    params['min_samples_split'] = float(params['min_samples_split'])
                
            if ALGORITHM == 'xgboost' or \
               ALGORITHM == 'gbr' or \
               ALGORITHM == 'extratrees' or \
               ALGORITHM == 'rf':
                params['n_estimators'] = int(params['n_estimators'])
                params['max_depth'] = int(params['max_depth'])
            
            elif ALGORITHM == 'knn':
                params['n_neighbors'] = int(params['n_neighbors'])
                params['leaf_size'] = int(params['leaf_size'])
                params['p'] = int(params['p'])
            elif ALGORITHM == 'svr':
                params['degree'] = int(params['degree'])
                params['coef0'] = int(params['coef0'])
                params['C'] = int(params['C'])
                params['cache_size'] = int(params['cache_size'])
                
            model = regressors(ALGORITHM)
            model.set_params(**params)

        i = 0

        log(f'Training from {fold_no} to {kfold} folds ...')
        if LSTM_ENABLED:
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])

        for i in range(0, kfold):
            X_train = inputs[train_index]
            y_train = targets[train_index]
            try:
                X_test = inputs[test_index]
                y_test = targets[test_index]
            except IndexError:
                test_index = np.arange(test_index[0], len(inputs))
                X_test = inputs[test_index]
                y_test = targets[test_index]

            # Generate a print
            # log('------------------------------------------------------------------------')
            # log(f'Training for fold {fold_no} ...')

            # Learn
            if LSTM_ENABLED:
                model, early_stop = init_lstm(X, LSTM_PARAMS)
                model.fit(X_train, y_train,
                          epochs=LSTM_PARAMS['epochs'],
                          batch_size=LSTM_PARAMS['batch_size'],
                          verbose=0,
                          shuffle=False,
                          callbacks=[early_stop])
            else:
                model.fit(X_train, y_train.ravel())

            if RECURSIVE:
                # Store predicted values
                y_pred = np.zeros(len(y_test))
                # Recursive predictions
                for j in range(len(y_test)):
                    # Next inputs for prediction
                    if GET_LAGGED:
                        if j > 0:
                            X_test_final = np.concatenate(
                                [X_test[j], np.array([y_lag])])
                        else:
                            X_test_final = X_test[0]
                            if DATASET_NAME.find('ONS') != -1:
                                X_test = np.delete(X_test, 6, 1)
                            elif DATASET_NAME.find('isone') != -1:
                                X_test = np.delete(X_test, 8, 1)
                    else:
                        X_test_final = X_test[j]

                    # Predict
                    y_pred[j] = model.predict(
                        X_test_final.reshape(-1, X_test_final.shape[0]))
                    # Save prediction
                    y_lag = y_pred[j]
            else:
                y_pred = model.predict(X_test)

            # Save y_pred
            kfoldPred.append(y_pred)
            # Prepare the plot data
            rows = test_index
            # rows = test
            df2 = df.iloc[rows[0]:rows[-1]+1]
            # df2.reset_index(drop=True,inplace=True)
            # df = pd.to_datetime(df)
            df2 = pd.to_datetime(df2)
            y_pred = np.float64(y_pred)

            if PLOT:
                # fig.add_trace(go.Scatter(x=df2,
                #                         y=y_pred,
                #                         name='Predicted Load (fold='+str(i+1)+")",
                #                         mode='lines'))
                plt.plot(df2, y_pred, label=f'Predicted Load (fold={fold_no}')

            y_pred_train = model.predict(X_train)
            y_pred_train = np.float64(y_pred_train)
            r2train = r2_score(y_train, y_pred_train)
            
            # Fix shape
            if len(y_pred) > 1:
                y_pred = y_pred.ravel()
            if len(y_test) > 1:
                try:
                    y_test = y_test.ravel()
                except AttributeError:
                    y_test = y_test.values.ravel()
            
            r2test = r2_score(y_test, y_pred)

            # log("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
            # log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
            n = len(X_test)
            p = X_test.shape[1]
            adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
            # log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            # log("RMSE: %f" % (rmse))

            mae = mean_absolute_error(y_test, y_pred)
            mae_percent = maep(y_test, y_pred)
            # log("MAE: %f" % (mae))

            mape = mean_absolute_percentage_error(y_test, y_pred)
            smape = symmetric_mape(y_test, y_pred)

            # log("MAPE: %.2f%%" % (mape))
            # log("sMAPE: %.2f%%" % (smape))

            # if plot:
            #     fig2 = go.Figure()
            #     fig2.add_shape(dict(
            #                     type="line",
            #                     x0=math.floor(min(np.array(y_test))),
            #                     y0=math.floor(min(np.array(y_test))),
            #                     x1=math.ceil(max(np.array(y_test))),
            #                     y1=math.ceil(max(np.array(y_test)))))
            #     fig2.update_shapes(dict(xref='x', yref='y'))
            #     fig2.add_trace(go.Scatter(x=y_test.reshape(y_test.shape[0]),
            #                             y=y_pred,
            #                             name='Real price VS Predicted Price (fold='+str(i+1)+")",
            #                             mode='markers'))
            #     fig2.update_layout(title='Real vs Predicted price',
            #                     xaxis_title=f'Real Demand - {y.columns[0]}',
            #                     yaxis_title=f'Predicted Load - {y.columns[0]}')
            #     fig2.show()

            # Generate generalization metrics
            # scores = model.evaluate(X_test, y_test, verbose=0)

            results[r].r2train_per_fold.append(r2train)
            results[r].r2test_per_fold.append(r2test)
            results[r].r2testadj_per_fold.append(adjr2_score)
            results[r].rmse_per_fold.append(rmse)
            results[r].mae_per_fold.append(mae)
            results[r].maep_per_fold.append(mae_percent)
            results[r].mape_per_fold.append(mape)
            results[r].smape_per_fold.append(smape)            
            results[r].decomposition = MODE
            results[r].nmodes = NMODES
            results[r].algorithm = ALGORITHM            
            results[r].test_name = 'loadForecast_' + y.columns[0]
            

            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes
            if not LSTM_ENABLED:
                # Sliding window
                train_index = np.arange(
                    train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)
            else:
                # Expanding window
                train_index = np.arange(0, train_index[-1] + 1 + train_size + test_size)

            test_index = test_index + train_size + test_size

        if PLOT:
            plt.rcParams.update({'font.size': 14})
            # plt.legend()
            plt.show()
            plt.tight_layout()
            if BOXCOX:
                if SAVE_FIG:
                    plt.savefig(
                        path+f'/results/pdf/{MODE}_{y.columns[0]}_BoxCox_loadForecast_k-fold_crossvalidation.pdf')
            else:
                if SAVE_FIG:
                    plt.savefig(
                        path+f'/results/pdf/{MODE}_{y.columns[0]}_legend_loadForecast_k-fold_crossvalidation.pdf')

            # Calculate feature importances
            try:
                plotFeatureImportance(X, model)
            except:
                pass

        # Print the results: average per fold
        log(f"Model name: {type(model).__name__}")
        results[r].model_name.append(type(model).__name__)
        results[r].name.append(y.columns[0])
        results[r].model_params = model.get_params()
        results[r].duration = round(time.time() - start_time_loadForecast, 2)
        results[r].printResults()
        if not enable_nni and SAVE_JSON:
            results[r].saveResults(path)
        

    else:  # NOT CROSSVALIDATION
        log(f'Predict only the last {testSize*X.shape[0]/24} days')
        log(f'Prediction on decomposed part: {y.columns[0]}')
        # transform training data & save lambda value
        # y_boxcox, lambda_boxcox = stats.boxcox(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testSize, random_state=0, shuffle=False)

        if not enable_nni:
            # model = xgboost.XGBRegressor(
            #                              colsample_bytree=0.8,
            #                              gamma=0.3,
            #                              learning_rate=0.03,
            #                              max_depth=7,
            #                              min_child_weight=6.0,
            #                              n_estimators=1000,
            #                              reg_alpha=0.75,
            #                              reg_lambda=0.01,
            #                              subsample=0.95,
            #                              seed=42)

            # from tensorflow.keras.models import Sequential
            # from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten, TimeDistributed
            # from tensorflow.keras.callbacks import EarlyStopping

            # ann_model = Sequential()
            # ann_model.add(Dense(16))
            # ann_model.add(LeakyReLU(alpha=0.05))
            # # Adding the hidden layers
            # for i in range(8):
            #     ann_model.add(Dense(units = 16))
            #     ann_model.add(LeakyReLU(alpha=0.05))
            # # output layer
            # ann_model.add(Dense(units = 1))
            # # Compiling the ANN
            # model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
            # early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

            # regressors = list()
            # regressors.append(('xgboost', xgboost.XGBRegressor()))
            # regressors.append(('knn', KNeighborsRegressor()))
            # regressors.append(('cart', DecisionTreeRegressor()))
            # regressors.append(('rf', RandomForestRegressor()))
            # regressors.append(('svm', svm.SVR()))
            # regressors.append(('gbr', GradientBoostingRegressor()))
            # regressors.append(('sgd', linear_model.SGDRegressor()))
            # regressors.append(('bayes'  , linear_model.BayesianRidge()))
            # regressors.append(('lasso', linear_model.LassoLars()))
            # regressors.append(('ard', linear_model.ARDRegression()))
            # regressors.append(('par', linear_model.PassiveAggressiveRegressor()))
            # regressors.append(('theilsen', linear_model.TheilSenRegressor()))
            # regressors.append(('linear', linear_model.LinearRegression()))

            # define meta learner model
            # meta_learner = linear_model.ARDRegression()

            # model.add(regressors)
            # model.add_meta(meta_learner)
            # model = StackingRegressor(estimators=regressors, final_estimator=meta_learner)
            # model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)

                # svm.SVR(kernel='poly',C=1)]
                # linear_model.SGDRegressor(),
                # linear_model.BayesianRidge(),
                # linear_model.LassoLars(),
                # linear_model.ARDRegression(),
                # linear_model.PassiveAggressiveRegressor(),
                # linear_model.TheilSenRegressor(),
                # linear_model.LinearRegression()]
            # model = GradientBoostingRegressor()
            model = regressors(ALGORITHM)

        else:  # nni enabled
            model = regressors(ALGORITHM)

        # for model in regressors:
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)

        # Prepare for plotting
        rows = X_test.index
        df2 = df.iloc[rows[0]:]

        if PLOT:
            # plt.figure()
            # plt.plot(df2,y_tested, color = 'red', label = 'Real data')
            plt.plot(df, y, label=f'Real data - {y.columns[0]}')
            plt.plot(df2, y_pred, label=f'Predicted data - {y.columns[0]}')
            if BOXCOX:
                plt.title(f'{DATASET_NAME} dataset Prediction - with BoxCox')
                plt.ylabel('Load [MW] - BoxCox')
            else:
                plt.title(f'{DATASET_NAME} dataset Prediction')
                plt.ylabel('Load [MW]')
            plt.xlabel('Date')
            plt.legend()
            if BOXCOX:
                if SAVE_FIG:
                    plt.savefig(
                        path+f'/results/pdf/{MODE}_{y.columns[0]}_noCV_BoxCox_pred_vs_real.pdf')
            else:
                if SAVE_FIG:            
                    plt.savefig(
                        path+f'/results/pdf/{MODE}_{y.columns[0]}_noCV_loadForecast_pred_vs_real.pdf')
            plt.show()
            plt.tight_layout()

        y_pred_train = model.predict(X_train)
        r2train = r2_score(y_train, y_pred_train)
        
        # Fix shape
        if len(y_pred) > 1:
            y_pred = y_pred.ravel()
        if len(y_test) > 1:
            try:
                y_test = y_test.ravel()
            except AttributeError:
                y_test = y_test.values.ravel()

        
        r2test = r2_score(y_test, y_pred)
        log(f"Model name: {type(model).__name__}")
        log("The R2 score on the Train set is:\t{:0.4f}".format(r2train))
        log("The R2 score on the Test set is:\t{:0.4f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
        log("The Adjusted R2 score on the Test set is:\t{:0.4f}".format(
            adjr2_score))

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        log("RMSE: %f" % (rmse))

        mae = mean_absolute_error(y_test, y_pred)
        log("MAE: %f" % (mae))


        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = symmetric_mape(y_test, y_pred)

        log("MAPE: %.2f%%" % (mape))
        log("sMAPE: %.2f%%" % (smape))

        # tscv = TimeSeriesSplit(n_splits=5)
        # scores = cross_val_score(model, X_, y_, cv=tscv, scoring='r2')
        # with np.printoptions(precision=4, suppress=True):
        #     log(scores)
        # log("Loss: {0:.4f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

        # Feature importance of XGBoost
        # if plot:
        #     ax = xgboost.plot_importance(model)
        #     ax.figure.set_size_inches(11,15)
        #     if dataset_name.find('ONS') != -1:
        #         ax.figure.savefig(path + f"/results/plot_importance_xgboost_{X_['SUBSYSTEM'].unique()[0]}.png")
        #     else:
        #         ax.figure.savefig(path + f"/results/plot_importance_xgboost_{dataset_name}.png")
        #     ax.figure.show()
        # log("\n--- \t{:0.4f} seconds --- Load Forecasting ".format(time.time() - start_time_loadForecast))

    log("\n--- \t{:0.4f} seconds --- Load Forecasting ".format(time.time() - start_time_loadForecast))
    return y_pred, testSize, kfoldPred, model


def composeSeasonal(decomposePred, model='stl-a'):
    if not CROSSVALIDATION:
        if model == 'stl-a':
            finalPred = sum(decomposePred)
        elif model == 'stl-m':
            finalPred = np.prod(decomposePred)
        elif model == 'emd' or model == 'eemd' or model == 'vmd' or model == 'ceemdan' or model == 'ewt':
            finalPred = sum(decomposePred)
        elif model == 'none':
            finalPred = decomposePred[0]
        elif model == 'robust-stl':
            finalPred = decomposePred[1] + decomposePred[2] + decomposePred[3]
    else:
        if model == 'none':
            finalPred = decomposePred[0]
        elif model == 'robust-stl':
            finalPred = decomposePred[1] + decomposePred[2] + decomposePred[3]
        elif model == 'stl-a':
            finalPred = [sum(x) for x in zip(*decomposePred)]
        elif model == 'stl-m':
            finalPred = []
            for x in zip(*decomposePred):
                fold = []
                for i in range(len(x[0])):
                    prod = x[0][i] * x[1][i] * x[2][i]
                    fold.append(prod)
                finalPred.append(np.array(fold))
        else:
            finalPred = [sum(x) for x in zip(*decomposePred)]
    return finalPred


def plotResults(X_, y_, y_pred, testSize, dataset_name='ONS'):
    start_time = time.time()
    
    if len(df) != len(y_):
        y_ = y_[:len(df)]
    if len(df) != len(X_):
        X_ = X_[:len(df)]

    if not CROSSVALIDATION:
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(y_pred.shape[0])
            elif y_pred.shape[0] == 1:
                y_pred = y_pred.reshape(y_pred.shape[1])
        if dataset_name.find('ONS') != -1:
            try:
                y_ = y_.drop(["SUBSYSTEM"], axis=1)
            except (AttributeError, KeyError) as e:
                pass

        X_train, X_test, y_train, y_test = train_test_split(
            X_, y_, test_size=testSize, random_state=0, shuffle=False)
        # Prepare for plotting
        rows = X_test.index
        df2 = df.iloc[rows[0]:]

        if PLOT:
            plt.figure()
            #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
            try:
                plt.plot(df, y_, label=f'Real data - {y_.columns[0]}')
                plt.plot(
                    df2, y_pred, label=f'Predicted data - {y_.columns[0]}')
            # except AttributeError:
            #     plt.plot(df,y_, label = f'Real data - {y_.name}')
            #     plt.plot(df2,y_pred, label = f'Predicted data - {y_.name}')
            except AttributeError:
                plt.plot(df, y_, label=f'Real data')
                plt.plot(df2, y_pred, label=f'Predicted data')
            plt.title(f'{DATASET_NAME} dataset Prediction')
            plt.xlabel('Date')
            plt.ylabel('Load [MW]')
            plt.legend()
            if SAVE_FIG:
                plt.savefig(path+f'/results/pdf/{MODE}_noCV_composed_pred_vs_real.pdf')
            plt.show()
            plt.tight_layout()
            
        # Fix shape
        if len(y_pred) > 1:
            y_pred = y_pred.ravel()
        if len(y_test) > 1:
            try:
                y_test = y_test.ravel()
            except AttributeError:
                y_test = y_test.values.ravel()

        r2test = r2_score(y_test, y_pred)
        log(f"Model name: {type(model).__name__}")
        log("The R2 score on the Test set is:\t{:0.4f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
        log("The Adjusted R2 score on the Test set is:\t{:0.4f}".format(
            adjr2_score))

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # log("RMSE: %f" % (rmse))

        mae = mean_absolute_error(y_test, y_pred)
        # log("MAE: %f" % (mae))
        
        mae_percent = maep(y_test, y_pred)
        # log("MAEP: %f" % (mae_percent))
                

        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = symmetric_mape(y_test, y_pred)
        # log("MAPE: %.2f%%" % (mape))
        # log("sMAPE: %.2f%%" % (smape))
        finalResults[0].r2train_per_fold.append(0)
        finalResults[0].r2test_per_fold.append(r2test)
        finalResults[0].r2testadj_per_fold.append(adjr2_score)
        finalResults[0].rmse_per_fold.append(rmse)
        finalResults[0].mae_per_fold.append(mae)
        finalResults[0].maep_per_fold.append(mae_percent)
        finalResults[0].mape_per_fold.append(mape)
        finalResults[0].smape_per_fold.append(smape)
        finalResults[0].name.append("DEMAND")
        finalResults[0].model_name.append(type(model).__name__)
        finalResults[0].model_params = model.get_params()
        finalResults[0].decomposition = MODE
        finalResults[0].nmodes = NMODES
        finalResults[0].algorithm = ALGORITHM
        finalResults[0].test_name = 'plotResults'
        finalResults[0].duration = round(time.time() - start_time, 2)
        

        finalResults[0].printResults()
        if not enable_nni and SAVE_JSON:
            finalResults[0].saveResults(path)

    else: # CROSSVALIDATION
        # Add real data to PLOT
        if PLOT:
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=df,
            #                          y=y_.squeeze(),
            #                          name=f'Electricity Demand [MW]',
            #                          mode='lines'))
            # # Edit the layout
            # fig.update_layout(title=f'{DATASET_NAME} dataset Load Forecasting - Cross-Validation of {KFOLD}-fold',
            #                     xaxis_title='Date',
            #                     yaxis_title=f'Load [MW]'
            #                     )
            plt.figure()
            plt.title(
                f'{DATASET_NAME} dataset Load Forecasting - Cross-Validation of {KFOLD}-fold')
            plt.xlabel('Date')
            plt.ylabel('Load [MW]')
            plt.plot(df, y_.squeeze(), color='darkgray',
                     label=f'Electricity Demand [MW]')

        # Change variable name because of lazyness
        inputs = np.array(X_)
        targets = np.array(y_)

        # Rest fold number
        fold_no = 1

        # Forecast X days
        test_size = round(FORECASTDAYS*24)
        train_size = math.floor((len(inputs)/KFOLD) - test_size)

        # Offset on Forecast window
        # offset = test_size*3

        if OFFSET > 0:
            log(f'OFFSET has been set by {OFFSET/24} days')
            # test_size = round((X.shape[0]-OFFSET)/uniqueYears.size/12/2)
            test_size = round(FORECASTDAYS*24)
            train_size = math.floor(((len(inputs)-OFFSET)/KFOLD) - test_size)

        train_index = np.arange(0, train_size+OFFSET)
        test_index = np.arange(train_size+OFFSET, train_size+test_size+OFFSET)
        
        for i in range(0, KFOLD):
            X_train = inputs[train_index]
            y_train = targets[train_index]
            try:
                X_test = inputs[test_index]
                y_test = targets[test_index]
            except IndexError:
                test_index = np.arange(test_index[0], len(inputs))
                X_test = inputs[test_index]
                y_test = targets[test_index]

            # Prepare the plot data
            rows = test_index
            # rows = test
            df2 = df.iloc[rows[0]:rows[-1]+1]
            # df2.reset_index(drop=True,inplace=True)
            # df = pd.to_datetime(df)
            df2 = pd.to_datetime(df2)
            y_pred[i] = np.float64(y_pred[i])

            if PLOT:
                # fig.add_trace(go.Scatter(x=df2,
                #                         y=y_pred[i],
                #                         name='Predicted Load (fold='+str(i+1)+")",
                #                         mode='lines'))
                plt.plot(df2, y_pred[i], label=f'Predicted Load (fold={i})')
                
                
            # Fix shape
            if len(y_pred[i]) > 1:
                y_pred[i] = y_pred[i].ravel()
            if len(y_test) > 1:
                try:
                    y_test = y_test.ravel()
                except AttributeError:
                    y_test = y_test.values.ravel()

            r2test = r2_score(y_test, y_pred[i])
        #    log("The R2 score on the Train set is:\t{:0.4f}".format(r2train))
        #    log("The R2 score on the Test set is:\t{:0.4f}".format(r2test))
            n = len(X_test)
            p = X_test.shape[1]
            adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
        #    log("The Adjusted R2 score on the Test set is:\t{:0.4f}".format(adjr2_score))

            rmse = np.sqrt(mean_squared_error(y_test, y_pred[i]))
           # log("RMSE: %f" % (rmse))

            mae = mean_absolute_error(y_test, y_pred[i])
           # log("MAE: %f" % (mae))
            mae_percent = maep(y_test, y_pred[i])

            
            # MAPE and sMAPE
            mape = mean_absolute_percentage_error(y_test, y_pred[i])
            smape = symmetric_mape(y_test, y_pred[i])
        #    log("MAPE: %.2f%%" % (mape))
        #    log("sMAPE: %.2f%%" % (smape))

            finalResults[0].r2train_per_fold.append(0)
            finalResults[0].r2test_per_fold.append(r2test)
            finalResults[0].r2testadj_per_fold.append(adjr2_score)
            finalResults[0].rmse_per_fold.append(rmse)
            finalResults[0].mae_per_fold.append(mae)
            finalResults[0].maep_per_fold.append(mae_percent)
            finalResults[0].mape_per_fold.append(mape)
            finalResults[0].smape_per_fold.append(smape)
            finalResults[0].name.append(f'kfold_{i}')            
            finalResults[0].decomposition = MODE
            finalResults[0].nmodes = NMODES
            finalResults[0].algorithm = ALGORITHM
            finalResults[0].test_name = 'plotResults'
            
            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes
            # train_index = np.concatenate((train_index, test_index), axis=0)
            train_index = np.arange(
                train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)

            test_index = test_index + train_size + test_size

        if PLOT:
            # fig.update_layout(
            #     font=dict(size=12),
            #     legend=dict(
            #     yanchor="top",
            #     y=0.99,
            #     xanchor="left",
            #     x=0.01,
            #     font=dict(
            #     size=12)
            # ))
            # fig.show()
            # if SAVE_FIG:
                # fig.write_image(file=path+'/results/pdf/loadForecast_k-fold_crossvalidation.pdf')
            plt.rcParams.update({'font.size': 14})
            plt.show()
            plt.tight_layout()
            if SAVE_FIG:
                plt.savefig(
                    path+f'/results/pdf/{MODE}_loadForecast_k-fold_crossvalidation.pdf')

        # Print the results: average per fold
        log(f"Model name: {type(model).__name__}")
        finalResults[0].model_name.append(type(model).__name__)
        finalResults[0].model_params = model.get_params()
        finalResults[0].duration = round(time.time() - start_time, 2)
        finalResults[0].printResults()
        if not enable_nni and SAVE_JSON:
            finalResults[0].saveResults(path)


def test_stationarity(data):
    from statsmodels.tsa.stattools import adfuller
    log('Stationarity test using Augmented Dickey-Fuller unit root test.')
    test_result = adfuller(
        data.iloc[:, 0].values, regression='ct', maxlag=360, autolag='t-stat')
    log('ADF Statistic: {}'.format(test_result[0]))
    log('p-value: {}'.format(test_result[1]))
    pvalue = test_result[1]
    log('Critical Values:')
    for key, value in test_result[4].items():
        log('\t{}: {}'.format(key, value))
    log(f'Used lags: {test_result[2]}')
    log(f'Number of observations: {test_result[3]}')

    if pvalue < 0.05:
        log(f'p-value < 0.05, so the series is stationary.')
    else:
        log(f'p-value > 0.05, so the series is non-stationary.')


def fast_fourier_transform(y_):
    from scipy.fft import fft, fftfreq
    if DATASET_NAME.find("ONS") != -1:
        for inputs in y_:
            xseries = inputs.drop(['SUBSYSTEM'], axis=1).values
            if xseries.shape[1] == 1:
                xseries = xseries.reshape(xseries.shape[0])
            # Number of sample points
            N = len(xseries)
            # sample spacing
            T = 1.0 / N
            yf = fft(xseries)
            xf = fftfreq(N, T)[:N//2]
            plt.figure()
            plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
            plt.ylim(0, max(2.0/N * np.abs(yf[0:N//2])))
            plt.show()
            plt.tight_layout()
    else:
        xseries = np.array(y_)
        if xseries.shape[1] == 1:
            xseries = xseries.reshape(xseries.shape[0])
        # Number of sample points
        N = len(np.array(xseries))
        # sample spacing
        T = 1.0 / N
        yf = fft(np.array(xseries))
        xf = fftfreq(N, T)[:N//2]
        plt.figure()
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.ylim(0, max(2.0/N * np.abs(yf[0:N//2])))
        plt.show()
        plt.tight_layout()


def emd_decompose(y_, Nmodes=3, dataset_name='ONS', mode='eemd', final_test=False):
    if mode == 'emd':
        printName = 'Empirical Mode Decomposition (EMD)'
    elif mode == 'eemd':
        printName = 'Ensemble Empirical Mode Decomposition (EEMD)'
    elif mode == 'vmd':
        printName = 'Variational Mode Decomposition (VMD)'
    elif mode == 'ceemdan':
        printName = 'Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)'
    elif mode == 'ewt':
        printName = 'Empirical Wavelet Transform (EWT)'
    log(f"{printName} has been started")
    tic = time.time()

    def do_emd():
        emd = EMD()
        # 4 years
        if DATASET_NAME.find('isone') != -1:
            emd.FIXE_H = 8
            emd.nbsym = 6
        elif DATASET_NAME.find('ONS') != -1:
            emd.FIXE_H = 1
            emd.nbsym = 2
        # 1 year
        # emd.FIXE = 1
        # emd.FIXE_H = 1
        # emd.nbsym = 1
        emd.spline_kind = 'cubic'
        IMFs = emd.emd(y_series, max_imf=Nmodes)
        return IMFs

    def do_eemd():
        if LOAD_DECOMPOSED:            
            # if GET_LAGGED:
            #     all_files = glob.glob(
            #         path + r"/datasets/" + DATASET_NAME + "/custom/" + f"eemd-{NMODES}_LAG_IMF*_{selectDatasets[0]}-{selectDatasets[-2]}.csv")
            # else:
            if not final_test:
                all_files = glob.glob(
                    path + r"/datasets/" + DATASET_NAME + r"/custom/" + f"eemd-{NMODES}_IMF_*_{selectDatasets[0]}-{selectDatasets[-2]}.csv")
            else:
                all_files = glob.glob(
                    path + r"/datasets/" + DATASET_NAME + r"/custom/" + f"eemd-{NMODES}_IMF_*_{selectDatasets[-1]}.csv")
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=None)
                    df = df.values.ravel()
                    IMFs.append(df)
            log("EEMD was successfully loaded.")
        else:
            eemd = EEMD(trials=100, noise_width=0.15, DTYPE=np.float16)
            eemd.MAX_ITERATION = 2000
            eemd.noise_seed(42)
            IMFs = eemd(y_series, max_imf=Nmodes)
        return IMFs

    def do_vmd():
        # VMD parameters
        alpha = 2000  # % moderate bandwidth constraint
        tau = 0  # % noise-tolerance (no strict fidelity enforcement)
        init = 1  # % initialize omegas uniformly
        tol = 1e-7
        DC = np.mean(y_series)   # no DC part imposed
        IMFs = VMD(y_series, alpha, tau, Nmodes, DC, init, tol)
        return IMFs

    def do_ceemdan():
        if LOAD_DECOMPOSED:
            # if GET_LAGGED:
            #     all_files = glob.glob(
            #         path + r"/datasets/" + DATASET_NAME + r"/custom/" + f"ceemdan-{NMODES}_LAG_IMF*_{selectDatasets[0]}-{selectDatasets[-1]}.csv")
            # else:
            if not final_test:
                path_composed = path + r"/datasets/" + DATASET_NAME + r"/custom/" + f"ceemdan-{NMODES}_IMF_*_{selectDatasets[0]}-{selectDatasets[-2]}*"
            else:
                path_composed = path + r"/datasets/" + DATASET_NAME + r"/custom/" + f"ceemdan-{NMODES}_IMF_*_{selectDatasets[-1]}*"
            
            all_files = glob.glob(path_composed)
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=None)
                    df = df.values.ravel()
                    IMFs.append(df)
            log("CEEMDAN was successfully loaded.")
        else:
            # CEEMDAN - Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
            ceemdan = CEEMDAN(trials=100, epsilon=0.01)
            ceemdan.noise_seed(42)
            IMFs = ceemdan(y_series, max_imf=Nmodes)
        return IMFs

    def do_ewt():
        # EWT - Empirical Wavelet Transform
        FFTreg = 'average'
        FFTregLen = 200
        gaussSigma = 15
        ewt, _, _ = ewtpy.EWT1D(y_series, N=Nmodes, log=0,
                                detect="locmax",
                                completion=0,
                                reg=FFTreg,
                                lengthFilter=FFTregLen,
                                sigmaFilter=gaussSigma)
        IMFs = []
        for i in range(ewt.shape[1]):
            IMFs.append(ewt[:, i])
        return IMFs

    y_series = np.array(y_)
    try:
        if y_series.shape[0] == 1:
            y_series = y_series.reshape(y_series.shape[1])
        elif y_series.shape[1] == 1:
            y_series = y_series.reshape(y_series.shape[0])
    except IndexError:
        pass
    if mode == 'emd':
        IMFs = do_emd()
    elif mode == 'eemd':
        IMFs = do_eemd()
    elif mode == 'vmd':
        IMFs = do_vmd()
    elif mode == 'ceemdan':
        IMFs = do_ceemdan()
    elif mode == 'ewt':
        IMFs = do_ewt()

    toc = time.time()
    log(f"{toc-tic:0.3f} seconds - {printName} has finished.")
    series_IMFs = []
    for i in range(len(IMFs)):
        series_IMFs.append(pd.DataFrame({f"IMF_{i}": IMFs[i]}))
    return series_IMFs


def get_lags_steps(X_test, y_train, steps_behind=24*7, steps_head=24*1):
    # log("Fetching the same period of prediction series...")
    # Take demand 1 week before of the forecast horizon
    lag_index = X_test.index.values-steps_behind
    
    # for the first y_train values, need to access part of y_trainset
    if lag_index[0] < 0:
        y_lag = y_trainset[-steps_behind:]
    else: # normal workflow
        try:
            y_lag = y_train.loc[X_test.index.values-steps_behind]
        except AttributeError:
            y_lag = y_train[X_test.index.values-steps_behind]

    # Rename column
    try:
        label = y_lag.columns[0]
    except AttributeError:
        y_lag = pd.DataFrame(y_lag)
        label = y_lag.columns[0]
    try:
        y_lag = y_lag.rename(columns={label: 'LAG'})
    except TypeError:
        y_lag = pd.DataFrame({'LAG': y_lag.ravel()})
    
    return y_lag
    

def plot_histogram(y_, xlabel):
    if PLOT:
        plt.figure()
        plt.title(f'{DATASET_NAME} Demand Histogram')
        plt.ylabel("Occurrences")
        if xlabel is not None:
            plt.xlabel(xlabel)
        sns.histplot(y_)
        plt.legend()
        plt.tight_layout()
        if xlabel.find('Box') != -1:
            if SAVE_FIG:
                plt.savefig(path+f'/results/pdf/{DATASET_NAME}_BoxCox_histogram.pdf')
        else:
            if SAVE_FIG:
                plt.savefig(path+f'/results/pdf/{DATASET_NAME}_demand_histogram.pdf')


def transform_stationary(y_, y_diff=0, invert=False):
    if invert and len(y_diff) > 1:
        try:
            result = np.cumsum(np.concatenate([y_.iloc[0], y_diff.ravel()]))
        except KeyError:
            result = np.cumsum(np.concatenate(
                [y_.iloc[0], y_diff.values.ravel()]))
        return result[1:]
    elif not invert:
        return pd.DataFrame(y_).diff()

    # Error
    if y_diff == 0:
        assert False


def get_lagged_y(X_, y_, n_steps=1):
    # log("Use lagged y (demand) to include as input in X")
    try:
        label = y_.columns[0]
    except AttributeError:
        y_ = pd.DataFrame(y_)
        label = y_.columns[0]
    y_lag = y_.shift(int(n_steps))

    try:
        y_lag = y_lag.rename(columns={label: 'DEMAND_LAG'})
    except TypeError:
        y_lag = pd.DataFrame({'DEMAND_LAG': y_lag.ravel()})
    concatlist = [X_, y_lag]
    X_ = pd.concat(concatlist, axis=1)
    # Drop null/NaN values
    # First save indexes to drop in y
    drop = X_[X_['DEMAND_LAG'].isnull()].index.values
    # Drop X
    X_ = X_.dropna().reset_index(drop=True)
    # Drop y
    try:
        y_ = y_.drop(index=drop).reset_index(drop=True)
    except KeyError:
        pass
    return X_, y_


def data_cleaning_columns(X, y):
    # Drop subsystem and date columns
    if DATASET_NAME.find('ONS') != -1:
        try:
            X = X.drop(['SUBSYSTEM', 'DATE'], axis=1)
        except KeyError:
            X = X.drop(['DATE'], axis=1)
    elif DATASET_NAME.find('isone') != -1:
        try:
            X = X.drop(['DATE'], axis=1)
        except KeyError:
            pass  # ignore it
    try:
        if y.columns.str.find("SUBSYSTEM") != -1 and y.columns[0] is not None:
            y = y.drop(['SUBSYSTEM'], axis=1)
        else:
            pass
    except AttributeError:
        pass

    return X, y


def finalTest(model, X_testset, y_testset, X_all, y_all, n_steps=STEPS_AHEAD, previous_models=PREVIOUS):
    start_time = time.time()
    
    if not FINAL_TEST:
        return
    log(f"Final test with test data - Forecast {int(n_steps/24)} day(s)")
    global df
    # if len(df) != len(y_all):
    #     # y_all = y_all[:len(df)]
    #     y_all = y_all[1:]
    # if len(df) != len(X_all):
    #     # X_all = X_all[:len(df)]
    #     X_all = X_all.drop(index=0).reset_index(drop=True)

    X_testset_copy = X_testset
    y_testset_copy = y_testset
    # Sanity check and drop unused columns
    X_testset, y_testset = data_cleaning_columns(X_testset, y_testset)
    X_all, y_all = data_cleaning_columns(X_all, y_all)

    # Drop date column on X_all
    # X_all = X_all.drop('DATE', axis=1)

    # Limit the horizon by n_steps
    # X_testset = X_testset[:n_steps]
    # y_testset = y_testset[:n_steps]
    #### This is for get_lagged TRUE ####
    # if GET_LAGGED:
        # index_shifting = index_shifting = X_test.reset_index()['index'] - 1
        # X_test = X_test.set_index(index_shifting)
        # y_test = pd.DataFrame(y_test).set_index(index_shifting)

    # y_all = pd.concat([y_,y_test], axis=1)
    # y_all = np.concatenate([y_all, y_test.values.ravel()])
    # X_all = pd.concat([X_, X_test], axis=0)

    # Normalize the signal
    y_transf, lambda_boxcox, sc1, minmax, min_y = data_transformation(y_testset)
    # y_transf, lambda_boxcox, sc1, minmax, min_y = data_transformation(y_all)

    # Data decompose
    y_decomposed_list = decomposeSeasonal(
        X_testset_copy, y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE, final_test=True)
    
    # Save decomposed signal
    saveDecomposedIMFs(y_decomposed_list, years=selectDatasets[-1])

    
    # Add real data to PLOT
    # if PLOT:
    #     if BOXCOX:
    #         plt.title(
    #             f'Electricity Prediction [MW] - with Box-Cox Transformation - {y.columns[0]}')
    #         plt.ylabel(f'Load [MW] - Box-Cox')
    #     else:
    #         plt.title(f'Electricity Prediction [MW] - {y.columns[0]}')
    #         plt.ylabel(f'Load [MW] - {y.columns[0]}')
    #     plt.xlabel(f'Date')
    #     plt.plot(df, y.squeeze(), color='darkgray', label='Real data')
    
    # List of predictions (IMF_0, IMF_1, ...)

    decomposePred = []
    kfoldPred = []
    save_test_index = []
    save_index = True
    
    # train and validation
    test_size = round(n_steps)
    train_size = math.floor((len(X_testset)/FINAL_FOLD) - test_size)
    if previous_models:
        raise 
    else: # previous_models = false
        for y_decomposed in y_decomposed_list:
            kfoldPred = []
            # Indexes
            train_index = np.arange(0, train_size)
            test_index = np.arange(train_size, train_size+test_size)
            if PLOT:
                plt.figure()
                plt.plot(y_decomposed, color='darkgray', label='Real data')
            for i in range(0, FINAL_FOLD):
                # Set indexes - Sliding window
                X_test = X_testset.iloc[test_index]
                y_test = y_decomposed.iloc[test_index]                
                X_train = X_testset.iloc[train_index]
                y_train = y_decomposed.iloc[train_index]
                
                if GET_LAGGED:
                    ## Include LAG values in X_test and X_train
                    y_lag = get_lags_steps(X_train, y_testset)
                    X_train = pd.concat([X_train, y_lag], axis=1)
                    y_lag = get_lags_steps(X_test, y_train)
                    X_test = pd.concat([X_test, y_lag], axis=1)
                    ####
                
                model = regressors(ALGORITHM)
                # Choose one model for each IMF
                if MULTIMODEL and MODE != 'none':
                    if y_decomposed.columns[0].find('IMF_') != -1 or \
                       y_decomposed.columns[0].find('Trend') != -1 or \
                       y_decomposed.columns[0].find('Residual') != -1 or \
                       y_decomposed.columns[0].find('Seasonal') != -1:
                        model = regressors(ALGORITHM)
                        local_params = open_json(model, ALGORITHM, y_decomposed.columns[0])
                else: # for individual algorithm Manual tuning                
                    local_params = open_json(model, ALGORITHM, 'none', manual=True)
                # Set model hyperparameters from json file
                model.set_params(**local_params)
                
                model.fit(X_train, y_train.values.ravel())

                # Store predicted values
                y_pred = model.predict(X_test)
                kfoldPred.append(y_pred)
                # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                # log(f"r2score = {r2_score(y_test, y_pred)}")
                # log(f"rmse = {rmse}")
                    
                # Increase indexes
                if not LSTM_ENABLED:
                    # Sliding window
                    train_index = np.arange(
                        train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)
                else:
                    # Expanding window
                    train_index = np.arange(0, train_index[-1] + 1 + train_size + test_size)
                
                if save_index:
                    save_test_index.append(test_index)
                
                test_index = test_index + train_size + test_size
            
            # To save test_index and avoid many loops doing the same thing
            save_index = False
            # Append predictions for each IMF
            decomposePred.append(kfoldPred)
        
    # Compose the signal
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)

    # Invert normalization
    y_composed = data_transformation_inverse(
        y_composed, lambda_boxcox, sc1, minmax, min_y, cv=True)

    ########################
    ### Evaluate results ###
    ########################
    # Font size
    FONT_SIZE = 18

    # Change y variable
    y_final = y_composed

    # Crop y_test
    ### wtf is that - maybe for GET_LAGGED
    #y_test = y_test[:-1]

    df2 = X_testset_copy['DATE']
    ## if GET_LAGGED:
        ## df2 = pd.DataFrame(df2).set_index(X_test.index.values[:-1])
        # df2 = pd.DataFrame(df2).set_index(X_test.index.values[:-1] - 24*FORECASTDAYS+1)

    df = pd.concat([df, df2], axis=0)

    # y_all = y_all[:len(y_)+len(y_final)]
    # y_all = y_all[:-24*FORECASTDAYS]
    
    
    # y_testset = np.array(y_testset).squeeze()
    y_final = np.array(y_final).squeeze()
    
    if PLOT:
        plt.figure()
        # Real data
        plt.plot(X_testset_copy['DATE'], y_testset, color='darkgray', label=f'Real data')
        plt.xlabel('Date', fontsize=FONT_SIZE)
        plt.ylabel('Load [MW]', fontsize=FONT_SIZE)
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE)
        
    
    y_test_list = []
    results = Results()
    
    # Predicted data
    for i in range(0, FINAL_FOLD):
        # Select correct range of y_testset for each fold
        y_test = y_testset[save_test_index[i]]
        # Save it to use later
        y_test_list.append(y_test)
        
        if PLOT:
            plt.plot(X_testset_copy['DATE'].iloc[save_test_index[i]], y_final[i], label=f'Forecasted', linestyle='--')
    
        # Fix shape
        if len(y_final[i]) > 1:
            y_final[i] = y_final[i].ravel()
        if len(y_test) > 1:
            try:
                y_test = y_test.ravel()
            except AttributeError:
                y_test = y_test.values.ravel()    
        
        
        r2test = r2_score(y_test, y_final[i])
        # log(f"Model name: {type(model).__name__}")
        # log("The R2 score on the Test set is:\t{:0.4f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
        # log("The Adjusted R2 score on the Test set is:\t{:0.4f}".format(
        #     adjr2_score))

        rmse = np.sqrt(mean_squared_error(y_test, y_final[i]))
        # log("RMSE: %f" % (rmse))

        mae = mean_absolute_error(y_test, y_final[i])
        # log("MAE: %f" % (mae))
        
        mae_percent = maep(y_test, y_final[i])
        # log("MAEP: %.3f%%" % (mae_percent))


        mape = mean_absolute_percentage_error(y_test, y_final[i])
        smape = symmetric_mape(y_test, y_final[i])
        # log("MAPE: %.3f%%" % (mape))
        # log("sMAPE: %.3f%%" % (smape))
        
        ###### Save results ########
        results.r2train_per_fold.append(0)
        results.r2test_per_fold.append(r2test)
        results.r2testadj_per_fold.append(adjr2_score)
        results.rmse_per_fold.append(rmse)
        results.mae_per_fold.append(mae)
        results.maep_per_fold.append(mae_percent)
        results.mape_per_fold.append(mape)
        results.smape_per_fold.append(smape)
        results.name.append('none')
        results.model_name.append(type(model).__name__)
        results.model_params = model.get_params()
        results.decomposition = MODE
        results.nmodes = NMODES
        results.algorithm = ALGORITHM
        results.test_name = 'finalTest'
        results.duration = round(time.time() - start_time, 2)
    

        
    if PLOT:
        if SAVE_FIG:
            plt.savefig(path+f'/results/pdf/{MODE}_noCV_composed_pred_vs_real.pdf')
        plt.show()
        plt.tight_layout()
    # Print results
    # Print the results: average per fold
    log(f"Model name: {type(model).__name__}")
    results.model_name.append(type(model).__name__)
    # results.name.append(y.columns[0])
    results.model_params = model.get_params()
    results.duration = round(time.time() - start_time, 2)
    results.printResults()
    
    # savePredictions(y_test_list, y_final)

    if not enable_nni and SAVE_JSON:
        results.saveResults(path)
        
        

    #################################
    
    if PLOT and False:
        plt.figure()
        plt.title(f'{DATASET_NAME} dataset Prediction - n-steps ahead')
        plt.xlabel('Time [h]')
        plt.ylabel('RMSE')
        xaxis = np.arange(0, n_steps, n_steps/10)

        if False:
            r2scores = []
            for i in range(len(y_final)):
                # if i == 0:
                r2test = 1-abs(y_test[i]-y_final[i])/y_test[i]
                # else:
                #r2test = r2_score(y_test[:i], y_final[:i])
                # r2test = calc_r2score(y_test[:i],y_final[:i])
                r2scores.append(r2test)

            plt.plot(r2scores)
            plt.xticks(xaxis)
            plt.show()
            plt.tight_layout()

        rmse = np.zeros(len(y_final))
        for i in range(len(y_final)):
            if i == 0:
                rmse[i] = abs(y_test[i] - y_final[i])
            else:
                rmse[i] = np.sqrt(mean_squared_error(
                    y_test[:i+1], y_final[:i+1]))

        plt.plot(rmse)
        plt.xticks(xaxis)
        plt.show()
        plt.tight_layout()
        
   

def data_transformation(y):
    sc1 = None
    minmax = None
    lambda_boxcox = None
    log("Plot Histogram")
    # plot_histogram(y, xlabel='Load [MW]')
    min_y = min(y)
    if BOXCOX:        
        if min_y <= 0:
            log("Shift negative to positive values + offset 1")
            y_transf = y+abs(min_y)+1
        else:
            y_transf = y
        log("Box-Cox transformation")
        if len(y_transf.shape) > 1:
            y_transf = y_transf.reshape(y_transf.shape[0])
        y_transf, lambda_boxcox = stats.boxcox(y_transf)
        y_transf = pd.DataFrame({'DEMAND': y_transf})
        log("Plot Histogram after Box-Cox Transformation")
        # plot_histogram(y_transf, xlabel='Box-Cox')
    else:
        y_transf = y
        try:
            y_transf = pd.DataFrame({'DEMAND': y_transf})
        except ValueError:
            y_transf = pd.DataFrame({'DEMAND': y_transf.ravel()})

    if STANDARDSCALER:
        label = y_transf.columns[0]
        sc1 = preprocessing.StandardScaler()
        y_transf = sc1.fit_transform(y_transf)
        try:
            y_transf = pd.DataFrame({label: y_transf})
        except ValueError:
            y_transf = pd.DataFrame({label: y_transf.ravel()})
        except AttributeError:
            y_transf = pd.DataFrame({label: y_transf.values.ravel()})
    if MINMAXSCALER:
        label = y_transf.columns[0]
        minmax = MinMaxScaler(feature_range=(-1, 1))
        y_transf = minmax.fit_transform(y_transf)
        try:
            y_transf = pd.DataFrame({label: y_transf})
        except ValueError:
            y_transf = pd.DataFrame({label: y_transf.ravel()})
        except AttributeError:
            y_transf = pd.DataFrame({label: y_transf.values.ravel()})

    return y_transf, lambda_boxcox, sc1, minmax, min_y


def data_transformation_inverse(y_composed, lambda_boxcox, sc1, minmax, min_y, cv):
    if cv:
        if MINMAXSCALER:
            log("Inverse MinMaxScaler transformation")
            for i in range(len(y_composed)):
                y_composed[i] = minmax.inverse_transform(
                    y_composed[i].reshape(y_composed[i].shape[0], 1))
        if STANDARDSCALER:
            log("Inverse StandardScaler transformation")
            for i in range(len(y_composed)):
                y_composed[i] = sc1.inverse_transform(
                    y_composed[i].reshape(y_composed[i].shape[0], 1))
        if BOXCOX:
            log("Inverse Box-Cox transformation")
            for i in range(len(y_composed)):
                y_composed[i] = special.inv_boxcox(
                    y_composed[i], lambda_boxcox)
                if min_y <= 0:
                    # log("Restore shifted values from positive to negative + offset -1")
                    y_composed[i] = y_composed[i] - abs(min_y)-1
    else:
        if MINMAXSCALER:
            log("Inverse MinMaxScaler transformation")
            try:
                y_composed = minmax.inverse_transform(
                    y_composed.reshape(y_composed.shape[0], 1))
            except AttributeError:
                y_composed = minmax.inverse_transform(
                    np.array(y_composed).reshape(np.array(y_composed).shape[0], 1))
        if STANDARDSCALER:
            log("Inverse StandardScaler transformation")
            try:
                y_composed = sc1.inverse_transform(
                    y_composed.reshape(y_composed.shape[0], 1))
            except AttributeError:
                y_composed = sc1.inverse_transform(
                    np.array(y_composed).reshape(np.array(y_composed).shape[0], 1))
        if BOXCOX:
            log("Inverse Box-Cox transformation")
            y_composed = special.inv_boxcox(y_composed, lambda_boxcox)
            if min_y <= 0:
                log("Restore shifted values from positive to negative + offset -1")
                y_composed = y_composed - abs(min_y)-1

    return y_composed

def plotFeatureImportance(X, model):
    # Font size
    FONT_SIZE = 18
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
    # plt.rcParams['font.size'] = '22'
    # plt.rcParams.update({'font.size': 22})
    # matplotlib.rcParams.update({'font.size': 22})
    # plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], height=0.2, align='center')
    # plt.axvline(x=0.03)
    # plt.rc('font', size=18)
    plt.yticks(range(len(indices)), list(names), fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.xlabel('Relative Importance', fontsize=FONT_SIZE)
    plt.show()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(path+f'/results/pdf/{DATASET_NAME}_{MODE}_feature_importance.pdf')

    # featImportance = pd.concat([pd.DataFrame({'Features':names}),
    #                  pd.DataFrame({'Relative_Importance':importances[indices]})], axis=1, sort=False)


def open_json(model, algorithm, imf, manual=False):
    if not manual:
        filePath = path + \
            f'/src/params/{DATASET_NAME}_{algorithm.upper()}_params_{imf.upper()}_{MODE.upper()}.json'
    else:
        filePath = path + \
            f'/src/params/{DATASET_NAME}_{algorithm.upper()}_params_MANUAL.json'
        
    try:
        # Opening JSON file
        fp = open(filePath)
        local_params = json.load(fp)
        if manual:
            log("Manual hyperparameters loaded successfully.")
        else:
            log("Multimodel hyperparameters loaded successfully.")
    except (FileNotFoundError, OSError, IOError) as e:
        # log(f'Hyperparameters JSON file not found: {e}')
        # log(f'Use default params...')
        local_params = model.get_params()
        pass
    return local_params


def init_lstm(X, params):
    model = Sequential()
    model.add(LSTM(units=params['neurons_width'],
                    activation=params['activation'],
                    input_shape=[None, X.shape[1]],
                    kernel_initializer="he_normal")
                )
    if params['dropout']:
        model.add(Dropout(params['dropout_val']))
    # Adding the hidden layers
    for i in range(params['hidden_layers']):
        model.add(Dense(params['neurons_width'], activation=params['activation'],
                        kernel_initializer="he_normal"))
        if params['dropout']:
            model.add(Dropout(params['dropout_val']))
    # Adding the output layer
    model.add(Dense(1))
    # print(model.summary())
    # Include loss and optimizer functions
    model.compile(loss='mse', optimizer=params['optimizer'])
    early_stop = EarlyStopping(
         monitor='loss', mode='min', patience=4, verbose=0)

    # history_lstm_model = model.fit(X_train, y_train,
    #                         epochs=_epochs,
    #                         batch_size=_batch,
    #                         verbose=1,
    #                         shuffle=False,
    #                         callbacks = [early_stop])
    return model, early_stop


def saveDecomposedIMFs(y_decomposed_list, years):
    if not LOAD_DECOMPOSED and (MODE == 'eemd' or MODE == 'ceemdan'):
        for imf in y_decomposed_list:
            if type(imf) is not type(pd.DataFrame()):
                imf = pd.DataFrame({imf.name: imf.values})
            
            try:
                if type(years) != type(list()):
                    imf.to_csv(
                        path+f'/datasets/{DATASET_NAME}/custom/{MODE}-{NMODES}_{imf.columns[0]}_{years}.csv', index=None, header=False)                
                else:
                    imf.to_csv(
                        path+f'/datasets/{DATASET_NAME}/custom/{MODE}-{NMODES}_{imf.columns[0]}_{years[0]}-{years[-1]}.csv', index=None, header=False)                
            except (FileNotFoundError, ValueError, OSError, IOError) as e:
                log("Failed to save CSV after data Decomposition")
                log(e)
                raise

def savePredictions(y_test, y_pred):
    if type(y_pred) is not type(pd.DataFrame()):
        y_pred = pd.DataFrame(y_pred)
    if type(y_test) is not type(pd.DataFrame()):
        y_test = pd.DataFrame(y_test)
        
    y_pred = pd.concat([y_test, y_pred], axis=1)
    try:
        y_pred.to_csv(
                path+f'/datasets/{DATASET_NAME}/y_pred/y_pred_{ALGORITHM}_{MODE}-{NMODES}_forecast{STEPS_AHEAD}_{selectDatasets[0]}-{selectDatasets[-1]}.csv', index=None, header=['y_test','y_pred'])
    except (FileNotFoundError, ValueError, OSError, IOError) as e:
        log("Failed to save CSV for y_pred.")
        log(e)
        raise

################
# MAIN PROGRAM
################
# Verify arguments for program execution
if '-nni' in sys.argv:
    enable_nni = True
    PLOT = False
    SAVE_JSON = False
    SAVE_FIG = False
    HYPERPARAMETER_TUNING = True
    FINAL_TEST_ONLY = False
    MULTIMODEL = False
if '-imf' in sys.argv:
    HYPERPARAMETER_IMF = sys.argv[sys.argv.index('-imf') + 1].upper()
if '-mode' in sys.argv:
    MODE = sys.argv[sys.argv.index('-mode') + 1]
if '-algo' in sys.argv:
    ALGORITHM = sys.argv[sys.argv.index('-algo') + 1]
if '-dataset' in sys.argv:
    DATASET_NAME = sys.argv[sys.argv.index('-dataset') + 1].upper()
if '-kfold' in sys.argv:
    KFOLD = int(sys.argv[sys.argv.index('-kfold') + 1])
if '-fdays' in sys.argv:
    FORECASTDAYS = int(sys.argv[sys.argv.index('-fdays') + 1])
if '-nmodes' in sys.argv:
    NMODES = int(sys.argv[sys.argv.index('-nmodes') + 1])
if '-steps' in sys.argv:
    STEPS_AHEAD = int(sys.argv[sys.argv.index('-steps') + 1])
if '-fold' in sys.argv:
    FINAL_FOLD = int(sys.argv[sys.argv.index('-fold') + 1])
if '-seed' in sys.argv:
    SEED_VALUE = int(sys.argv[sys.argv.index('-seed') + 1])
if '-loop' in sys.argv:
    LOOP = True
    PLOT = False
if '-load' in sys.argv:
    LOAD_DECOMPOSE = True
if '-plotoff' in sys.argv:
    PLOT = False


log(f"Dataset: {DATASET_NAME}")
log(f"Years: {selectDatasets}")
log(f"CrossValidation: {CROSSVALIDATION}")
log(f"KFOLD: {KFOLD}")
log(f"OFFSET: {OFFSET}")
log(f"FORECASTDAYS: {FORECASTDAYS}")
log(f"NMODES: {NMODES}")
log(f"MODE: {MODE}")
log(f"BOXCOX: {BOXCOX}")
log(f"STANDARDSCALER: {STANDARDSCALER}")
log(f"MINMAXSCALER: {MINMAXSCALER}")
##############################

params = nni.get_next_parameter()
# Initial message
log("Time Series Regression - Load forecasting using ensemble algorithms")
# Dataset import
dataset = datasetImport(selectDatasets, dataset_name=DATASET_NAME)
# Data cleaning and set the input and reference data
X, y = dataCleaning(dataset, dataset_name=DATASET_NAME)
# Include new data
X, y = featureEngineering(dataset, X, y, selectDatasets, weekday=True, holiday=True,
                          holiday_bridge=False, demand_lag=True, dataset_name=DATASET_NAME)

# Redefine df
global df
df = X['DATE']
# Outlier removal
y = outlierCleaning(y, dataset_name=DATASET_NAME)

if PLOT and True:
    plt.figure()
    plt.title(f'{DATASET_NAME} dataset demand curve')
    plt.xlabel('Date')
    plt.ylabel('Load [MW]')
    plt.plot(df, y)
    plt.show()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(path+f'/results/pdf/{DATASET_NAME}_after_outlierClean.pdf')
# List of results
results = []
finalResults = []

# Initialize fig
fig = go.Figure()

list_IMFs = []
# fast_fourier_transform(y)

# Prediction list of different components of decomposition to be assemble in the end
decomposePred = []
listOfDecomposePred = []
models = []

##########################################
######### Starting the framework #########
##########################################

y_testset = y[(-24*365):]
X_testset = X[(-24*365):]
X_trainset = X[:(-24*365)]
y_trainset = y[:(-24*365)]

# Data transformation - BoxCox and MinMaxScaler/StandardScaler
y_transf, lambda_boxcox, sc1, minmax, min_y = data_transformation(y_trainset)

# Decompose data
y_decomposed_list = decomposeSeasonal(
    X_trainset, y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE, final_test=False)

# Save decomposed data
saveDecomposedIMFs(y_decomposed_list, years=selectDatasets[:-1])
# Ensure final test will need to decompose its part
# LOAD_DECOMPOSED = False

# Split the test data from training/validation data
# y_testset = y[(-24*365):]
# X_testset = X[(-24*365):]
# X_trainset = X[:(-24*365)]
# y_trainset = y[:(-24*365)]
# y_testset = y[(-24*TEST_DAYS):]
# X_testset = X[(-24*TEST_DAYS):]
# X_trainset = X[:(-24*TEST_DAYS)]
# y_trainset = y[:(-24*TEST_DAYS)]

# Reduce decomposed list to trainset only
#newlist = []
#for decompose in y_decomposed_list:
#    newlist.append(decompose[:-24*365])
#
#y_decomposed_list = newlist

df = X_trainset['DATE']
X_testset = X_testset.reset_index(drop=True)
# y_testset = y_testset.reset_index(drop=True)

# Index for Results
r = 0
models = []
decomposePred = []
results = []
if not FINAL_TEST_ONLY:
    # Loop over decomposed data
    for y_decomposed in y_decomposed_list:
        if type(y_decomposed) is not type(pd.DataFrame()):
            y_decomposed = pd.DataFrame({y_decomposed.name: y_decomposed.values})
        ######## This is for hyperparameter tuning #######
        if HYPERPARAMETER_TUNING:
            if y_decomposed.columns[0].find(HYPERPARAMETER_IMF) == -1:
                continue
        ##################################################

        results.append(Results())  # Start new Results instance every loop step

        # Load Forecasting
        y_out, testSize, kfoldPred, model = loadForecast(
            X=X_trainset, y=y_decomposed, CrossValidation=CROSSVALIDATION, kfold=KFOLD, offset=OFFSET, forecastDays=FORECASTDAYS, dataset_name=DATASET_NAME)
        # Save the current model for further usage
        models.append(model)

        if CROSSVALIDATION:
            decomposePred.append(kfoldPred)
        else:
            decomposePred.append(y_out)
        r += 1

    # Join decomposed values, invert transformations, perform final tests
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)
    data_transformation_inverse(
        y_composed, lambda_boxcox, sc1, minmax, min_y, cv=CROSSVALIDATION)
    log("Print and plot the results")
    finalResults.append(Results())
    plotResults(X_=X_trainset, y_=y, y_pred=y_composed,
                testSize=testSize, dataset_name=DATASET_NAME)
finalTest(model=models, X_testset=X_testset,
            y_testset=y_testset, X_all=X, y_all=y)

if enable_nni:
    log("Publish the results on AutoML nni")
    if HYPERPARAMETER_TUNING:
        rmsetestResults = results[0].rmse_per_fold
    else:        
        rmsetestResults = finalResults[0].rmse_per_fold
    rmseScoreAvg = np.mean(rmsetestResults)
    log(f"rmseScoreAvg = {rmseScoreAvg}")
    if not LOOP:
        nni.report_final_result(rmseScoreAvg)
    # results[0].printResults()


log("\n--- \t{:0.3f} seconds --- the end of the file.".format(time.time() - start_time))


# trend = pd.concat([df, y_decomposed_list[1]], axis=1)
# seasonal = pd.concat([df, y_decomposed_list[2]], axis=1)
# remainder = pd.concat([df, y_decomposed_list[3]], axis=1)
# trend.to_csv(path+f'/robust-stl_trend_{selectDatasets[0]}.csv', index = None, header=True)
# seasonal.to_csv(path+f'/robust-stl_seasonal_{selectDatasets[0]}.csv', index = None, header=True)
# remainder.to_csv(path+f'/robust-stl_remainder_{selectDatasets[0]}.csv', index = None, header=True)
#


# Close logging handlers to release the log file
handlers = logging.getLogger().handlers[:]
for handler in handlers:
    handler.close()
    logging.getLogger().removeHandler(handler)
