from tensorflow import set_random_seed
from numpy.random import seed
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
"""
Time-Series Decomposition
Author: Marcos Yamasaki
04/03/2021
"""
import time
import logging
from log import log
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
DATASET_NAME = 'ISONewEngland'
# Enable nni for AutoML
enable_nni = False
# Set True to plot curves
plot = True
# Configuration for Forecasting
CROSSVALIDATION = True
KFOLD = 1
OFFSET = 0
FORECASTDAYS = 7
NMODES = 6
MODE = 'eemd'
BOXCOX = True
STANDARDSCALER = True
MINMAXSCALER = False
DIFF = False
LOAD_DECOMPOSED = False
RECURSIVE = True
GET_LAGGED = True
PREVIOUS = True
HYPERPARAMETER_TUNING = False
HYPERPARAMETER_IMF = 'IMF_0'
STEPS_AHEAD = 24*1
TEST_DAYS = 29
MULTIMODEL = False
LSTM_ENABLED = False
FINAL_TEST = False
# Selection of year
selectDatasets = ["2015", "2016", "2017", "2018"]
# selectDatasets = ["2017","2018"]
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


# Seed Random Numbers with the TensorFlow Backend
seed(42)
set_random_seed(42)


def datasetImport(selectDatasets, dataset_name='ONS'):
    log('Dataset import has been started')
    # Save all files in the folder
    if dataset_name.find('ONS') != -1:
        filename = glob.glob(path + r'/datasets/ONS/*south*.csv')
        filename = filename[0].replace('\\', '/')
        dataset = pd.read_csv(filename, index_col=None,
                              header=0, delimiter=";")
        # Select only selected data
        datasetList = []
        for year in selectDatasets:
            datasetList.append(dataset[dataset['DATE'].str.find(year) != -1])
    elif dataset_name.find('ISONewEngland') != -1:
        all_files = glob.glob(
            path + r'/datasets/ISONewEngland/csv-fixed/*.csv')
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
    elif dataset_name.find('ISONewEngland') != -1:
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
    elif dataset_name.find('ISONewEngland') != -1:
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

    # elif dataset_name.find('ISONewEngland') != -1:
    #     X.append(X)
    #     y.append(y)

    return X, y


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def maep(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)
    

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


def decomposeSeasonal(X_, y_, dataset_name='ONS', Nmodes=3, mode='stl-a'):
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
        elif dataset_name.find('ISONewEngland') != -1:
            concatlist = [data, pd.DataFrame(y_)]
        data = pd.concat(concatlist, axis=1)

        data.reset_index(inplace=True)
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data.set_index('DATE')
        data = data.drop(['index'], axis=1)
        data.columns = ['DEMAND']
        if mode == 'stl-a':
            model = 'additive'
        elif mode == 'stl-m':
            model = 'multiplicative'
        result = seasonal_decompose(
            data, period=24, model=model, extrapolate_trend='freq')
        result.trend.reset_index(drop=True, inplace=True)
        result.seasonal.reset_index(drop=True, inplace=True)
        result.resid.reset_index(drop=True, inplace=True)
        result.observed.reset_index(drop=True, inplace=True)
        result.trend.name = 'Trend'
        result.seasonal.name = 'Seasonal'
        result.resid.name = 'Residual'
        result.observed.name = 'Observed'
        decomposeList = [result.trend, result.seasonal, result.resid]

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
            y_, Nmodes=Nmodes, dataset_name=DATASET_NAME, mode=mode)
    elif mode == 'robust-stl':
        labels = ['Observed', 'Trend', 'Seasonal', 'Remainder']
        if LOAD_DECOMPOSED:
            all_files = glob.glob(
                path + r'/datasets/ISONewEngland/custom/robust-stl*.csv')
            # Initialize dataset list
            decomposeList = []
            i = 0
            concat = []
            # Read all csv files and concat them
            for filename in all_files:
                if filename.find(MODE) != -1:
                    df = pd.read_csv(filename, index_col=None, header=0)
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
    if plot and False:
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
        fig.write_image(f"{path}{DATASET_NAME}_outliers_"+columnName+".pdf")

    # Fix outliers by removing and replacing with interpolation
    try:
        y_ = y_.replace([outliers], np.nan)
    except ValueError:
        y_ = y_.replace(outliers, np.nan)
    y_ = y_.interpolate(method='linear', axis=0).ffill().bfill()

    print('Outliers fixed: ', end='\n')
    print(y_.loc[outliers.index.values], end='\n')

    # Transform to numpy arrays
    y_ = np.array(y_)
    y_ = y_.reshape(y_.shape[0])

    if plot and False:
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
        fig.write_image(f"{DATASET_NAME}_outliers_fixed_"+columnName+".pdf")

    return y_


def loadForecast(X, y, CrossValidation=False, kfold=5, offset=0, forecastDays=15, dataset_name='ONS'):
    log("Load Forecasting algorithm has been started")
    start_time_xgboost = time.time()

    global df, fig
    kfoldPred = []
    # Plot
    if plot:
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

        # Add real data to plot
        if plot:
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
            # regressors.append(('rf', RandomForestRegressor(n_estimators=750,
            #                                               max_depth=32,
            #                                               min_samples_split=2,
            #                                               min_samples_leaf=1,
            #                                               max_features="auto",
            #                                               max_leaf_nodes=None,
            #                                               min_impurity_decrease=0.001,
            #                                               bootstrap=True,
            #                                               random_state=42,
            #                                               n_jobs=-1)))
            # regressors.append(
            #     ('svm', svm.SVR(kernel='rbf', gamma=0.001, C=10000)))
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
            # meta_learner = GradientBoostingRegressor() # 0.85873
            # meta_learner = ExtraTreesRegressor() # 0.85938
            # meta_learner = linear_model.TheilSenRegressor() # 0.87946
            # meta_learner = linear_model.ARDRegression()  # 0.88415
            # meta_learner = LinearRegression() # 0.88037
            # meta_learner = linear_model.BayesianRidge() # 0.877

            # model = VotingRegressor(estimators=regressors)
            # model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)
            # model = StackingRegressor(
            #     estimators=regressors, final_estimator=meta_learner)
            # model = GradientBoostingRegressor()

            # model = xgboost.XGBRegressor()
            model = GradientBoostingRegressor()

            if LSTM_ENABLED:
                # LSTM parameters
                _batch = 24
                _epochs = 50
                _neurons = 128
                _hidden_layers = 4
                _optimizer = 'Adam'
                _dropout = False
                _dropoutVal = 0.2
                _activation = LeakyReLU(alpha=0.2)
                lstm_params = {
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
                # model, early_stop = init_lstm(lstm_params)

            # Choose one model for each IMF
            if MULTIMODEL and MODE != 'none':
                if y.columns[0].find('IMF_0') != -1:
                    # model = ExtraTreesRegressor()
                    # model = xgboost.XGBRegressor()
                    # local_params = open_json(model, 'XGB', 'IMF_0')
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_0')
                elif y.columns[0].find('IMF_1') != -1:
                    # model = xgboost.XGBRegressor()
                    # local_params = open_json(model, 'XGB', 'IMF_1')
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_1')
                elif y.columns[0].find('IMF_2') != -1:
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_2')
                elif y.columns[0].find('IMF_3') != -1:
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_3')
                elif y.columns[0].find('IMF_4') != -1:
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_4')
                elif y.columns[0].find('IMF_5') != -1:
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_5')
                elif y.columns[0].find('IMF_6') != -1:
                    model = GradientBoostingRegressor()
                    local_params = open_json(model, 'GBR', 'IMF_6')
                model.set_params(**local_params)
        else:  # nni enabled
            # if params['warm_start'] == "True":
            #     warm_start = True
            # elif params['warm_start'] == "False":
            #     warm_start = False

            if params['min_samples_split'] > 1:
                params['min_samples_split'] = int(params['min_samples_split'])
            else:
                params['min_samples_split'] = float(params['min_samples_split'])
            
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])

            model = GradientBoostingRegressor()
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
                model, early_stop = init_lstm(X, lstm_params)
                model.fit(X_train, y_train,
                          epochs=lstm_params['epochs'],
                          batch_size=lstm_params['batch_size'],
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
                    if j > 0:
                        X_test_final = np.concatenate(
                            [X_test[j], np.array([y_lag])])
                    else:
                        X_test_final = X_test[0]
                        if DATASET_NAME.find('ONS') != -1:
                            X_test = np.delete(X_test, 6, 1)
                        elif DATASET_NAME.find('ISONewEngland') != -1:
                            X_test = np.delete(X_test, 8, 1)

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

            if plot:
                # fig.add_trace(go.Scatter(x=df2,
                #                         y=y_pred,
                #                         name='Predicted Load (fold='+str(i+1)+")",
                #                         mode='lines'))
                plt.plot(df2, y_pred, label=f'Predicted Load (fold={fold_no}')

            y_pred_train = model.predict(X_train)
            y_pred_train = np.float64(y_pred_train)
            r2train = r2_score(y_train, y_pred_train)
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

            # Fix shape
            if len(y_pred) > 1:
                y_pred = y_pred.ravel()
            if len(y_test) > 1:
                try:
                    y_test = y_test.ravel()
                except AttributeError:
                    y_test = y_test.values.ravel()

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
            results[r].name.append(y.columns[0])

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

        if plot:
            plt.rcParams.update({'font.size': 14})
            # plt.legend()
            plt.show()
            plt.tight_layout()
            if BOXCOX:
                plt.savefig(
                    path+f'/results/{MODE}_{y.columns[0]}_BoxCox_loadForecast_k-fold_crossvalidation.pdf')
            else:
                plt.savefig(
                    path+f'/results/{MODE}_{y.columns[0]}_legend_loadForecast_k-fold_crossvalidation.pdf')

            # Calculate feature importances
            try:
                plotFeatureImportance(X, model)
            except:
                pass

        # Print the results: average per fold
        results[r].printResults()

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
            model = GradientBoostingRegressor()

        else:  # mni enabled
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

        # for model in regressors:
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)

        # Prepare for plotting
        rows = X_test.index
        df2 = df.iloc[rows[0]:]

        if plot:
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
                plt.savefig(
                    path+f'/results/{MODE}_{y.columns[0]}_noCV_BoxCox_pred_vs_real.pdf')
            else:
                plt.savefig(
                    path+f'/results/{MODE}_{y.columns[0]}_noCV_loadForecast_pred_vs_real.pdf')
            plt.show()
            plt.tight_layout()

        y_pred_train = model.predict(X_train)
        r2train = r2_score(y_train, y_pred_train)
        r2test = r2_score(y_test, y_pred)
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

        # Fix shape
        if len(y_pred) > 1:
            y_pred = y_pred.ravel()
        if len(y_test) > 1:
            try:
                y_test = y_test.ravel()
            except AttributeError:
                y_test = y_test.values.ravel()

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
        # log("\n--- \t{:0.4f} seconds --- Load Forecasting ".format(time.time() - start_time_xgboost))

    log("\n--- \t{:0.4f} seconds --- Load Forecasting ".format(time.time() - start_time_xgboost))
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

        if plot:
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
            plt.savefig(path+f'/results/{MODE}_noCV_composed_pred_vs_real.pdf')
            plt.show()
            plt.tight_layout()

        r2test = r2_score(y_test, y_pred)
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
        
        mae_percent = maep(y_test, y_pred)
        log("MAEP: %f" % (mae_percent))
                

        # Fix shape
        if len(y_pred) > 1:
            y_pred = y_pred.ravel()
        if len(y_test) > 1:
            try:
                y_test = y_test.ravel()
            except AttributeError:
                y_test = y_test.values.ravel()
        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = symmetric_mape(y_test, y_pred)
        log("MAPE: %.2f%%" % (mape))
        log("sMAPE: %.2f%%" % (smape))
        finalResults[0].r2train_per_fold.append(0)
        finalResults[0].r2test_per_fold.append(r2test)
        finalResults[0].r2testadj_per_fold.append(adjr2_score)
        finalResults[0].rmse_per_fold.append(rmse)
        finalResults[0].mae_per_fold.append(mae)
        finalResults[0].maep_per_fold.append(mae_percent)
        finalResults[0].mape_per_fold.append(mape)
        finalResults[0].smape_per_fold.append(smape)
        finalResults[0].name.append("DEMAND")

        finalResults[0].printResults()

    else:
        # Add real data to plot
        if plot:
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
            finalResults.append(Results())
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

            if plot:
                # fig.add_trace(go.Scatter(x=df2,
                #                         y=y_pred[i],
                #                         name='Predicted Load (fold='+str(i+1)+")",
                #                         mode='lines'))
                plt.plot(df2, y_pred[i], label=f'Predicted Load (fold={i})')

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

            # Fix shape
            if len(y_pred[i]) > 1:
                y_pred[i] = y_pred[i].ravel()
            if len(y_test) > 1:
                try:
                    y_test = y_test.ravel()
                except AttributeError:
                    y_test = y_test.values.ravel()
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

            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes
            # train_index = np.concatenate((train_index, test_index), axis=0)
            train_index = np.arange(
                train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)

            test_index = test_index + train_size + test_size

        if plot:
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
            # fig.write_image(file=path+'/results/loadForecast_k-fold_crossvalidation.pdf')
            plt.rcParams.update({'font.size': 14})
            plt.show()
            plt.tight_layout()
            plt.savefig(
                path+f'/results/{MODE}_loadForecast_k-fold_crossvalidation.pdf')

        # Print the results: average per fold
        finalResults[0].printResults()


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


def emd_decompose(y_, Nmodes=3, dataset_name='ONS', mode='eemd'):
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
        if DATASET_NAME.find('ISONewEngland') != -1:
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
            all_files = glob.glob(
                path + r"/datasets/ISONewEngland/custom/" + f"eemd_IMF*_forecast{FORECASTDAYS}_{selectDatasets[0]}-{selectDatasets[-1]}.csv")
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=0)
                    df = df.values.ravel()
                    IMFs.append(df)

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
            all_files = glob.glob(
                path + r"/datasets/ISONewEngland/custom/" + f"ceemdan_IMF*_forecast{FORECASTDAYS}_{selectDatasets[0]}-{selectDatasets[-1]}.csv")
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=0)
                    df = df.values.ravel()
                    IMFs.append(df)
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


def get_training_set_for_same_period(X_train, y_train, X_test, y_test, forecastDays=15, dataset_name='ONS'):
    log("Fetching the same period of prediction series...")
    X_train[X_train['Month'] == X_test['Month']]
    X_train[X_train['Day'] == X_test['Day']]
    X_train[X_train['Hour'] == X_test['Hour']]


def plot_histogram(y_, xlabel):
    if plot:
        plt.figure()
        plt.title(f'{DATASET_NAME} Demand Histogram')
        plt.ylabel("Occurrences")
        if xlabel is not None:
            plt.xlabel(xlabel)
        sns.histplot(y_)
        plt.legend()
        plt.tight_layout()
        if xlabel.find('Box') != -1:
            plt.savefig(path+f'/results/{DATASET_NAME}_BoxCox_histogram.pdf')
        else:
            plt.savefig(path+f'/results/{DATASET_NAME}_demand_histogram.pdf')


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
    elif DATASET_NAME.find('ISONewEngland') != -1:
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


def finalTest(model, X_test, y_test, X_, y_, testSize, n_steps=STEPS_AHEAD, previous_models=PREVIOUS):
    if not FINAL_TEST:
        return
    log(f"Final test with test data - Forecast {int(n_steps/24)} day(s)")
    global df
    if len(df) != len(y_):
        # y_ = y_[:len(df)]
        y_ = y_[1:]
    if len(df) != len(X_):
        # X_ = X_[:len(df)]
        X_ = X_.drop(index=0).reset_index(drop=True)

    X_test_copy = X_test
    y_test_copy = y_test
    # Sanity check and drop unused columns
    X_test, y_test = data_cleaning_columns(X_test, y_test)

    # Drop date column on X_
    X_ = X_.drop('DATE', axis=1)

    # Limit the horizon by n_steps
    X_test = X_test[:n_steps+1]
    y_test = y_test[:n_steps+1]
    index_shifting = index_shifting = X_test.reset_index()['index'] - 1
    X_test = X_test.set_index(index_shifting)
    y_test = pd.DataFrame(y_test).set_index(index_shifting)

    # y_all = pd.concat([y_,y_test], axis=1)
    y_all = np.concatenate([y_, y_test.values.ravel()])
    X_all = pd.concat([X_, X_test], axis=0)

    # Normalize the signal
    y_transf, lambda_boxcox, sc1, minmax, min_y = data_transformation(y_)

    # Data decompose
    y_decomposed_list = decomposeSeasonal(
        df, y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE)

    # List of predictions (IMF_0, IMF_1, ...)
    decomposePred = []

    if previous_models:
        for (model, y_decomposed) in zip(models, y_decomposed_list):
            if GET_LAGGED:
                X_lagged, y_lagged = get_lagged_y(X_, y_decomposed, n_steps=1)

            # Store predicted values
            y_pred = np.zeros(n_steps)
            # Recursive predictions
            for i in range(n_steps):
                if GET_LAGGED:
                    if i > 0:
                        X_test_final = X_test.iloc[i].append(pd.Series(y_lag))
                    else:
                        X_test_final = X_test.iloc[0].append(
                            pd.Series(X_lagged['DEMAND_LAG'][0]))
                    # Rename
                    X_test_final = X_test_final.rename({0: 'DEMAND_LAG'})
                else:
                    X_test_final = X_test.iloc[i]

                # Predict
                if LSTM_ENABLED:
                    y_pred[i] = model.predict(
                        X_test_final.values.reshape(1, 1, X_test_final.shape[0]))
                else:
                    y_pred[i] = model.predict(
                        X_test_final.values.reshape(-1, X_test_final.shape[0]))
                # Save prediction
                y_lag = y_pred[i]

            decomposePred.append(y_pred)
    else:
        for y_decomposed in y_decomposed_list:
            train_size = round(len(X_)/(KFOLD))
            if GET_LAGGED:
                # X_all_lag, y_all_lag = get_lagged_y(X_all, y_decomposed, n_steps=1)
                X_lagged, y_lagged = get_lagged_y(X_, y_decomposed, n_steps=1)
                X_train = X_lagged[-train_size:]
                y_train = y_lagged[-train_size:]
            else:
                X_train = X_[-train_size:]
                y_train = y_decomposed[-train_size:]

            model = GradientBoostingRegressor()
            # model = xgboost.XGBRegressor()
            model.fit(X_train, y_train.values.ravel())

            # Store predicted values
            y_pred = np.zeros(n_steps)
            # Recursive predictions
            for i in range(n_steps):
                if GET_LAGGED:
                    if i > 0:
                        X_test_final = X_test.iloc[i].append(pd.Series(y_lag))
                    else:
                        X_test_final = X_test.iloc[0].append(
                            pd.Series(X_lagged['DEMAND_LAG'][0]))
                    # Rename
                    X_test_final = X_test_final.rename({0: 'DEMAND_LAG'})
                else:
                    # X_test_final = X_test[i:i+1]
                    X_test_final = X_test.iloc[i]
                # Predict
                try:
                    y_pred[i] = model.predict(X_test_final)
                except (ValueError, AttributeError) as e:
                    y_pred[i] = model.predict(
                    X_test_final.values.reshape(-1, X_test_final.shape[0]))
                    pass
                # Save prediction
                y_lag = y_pred[i]

            decomposePred.append(y_pred)

    # Compose the signal
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)

    # Invert normalization
    y_composed = data_transformation_inverse(
        y_composed, lambda_boxcox, sc1, minmax, min_y, cv=False)

    ########################
    ### Evaluate results ###
    ########################
    # Font size
    FONT_SIZE = 18

    # Change y variable
    y_final = y_composed

    # Crop y_test
    y_test = y_test[:-1]

    # Split original series into train and test data
    # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = testSize, random_state = 0, shuffle = False)
    # Prepare for plotting
    # rows = X_test.index
    # df2 = df.iloc[rows[0]:]
    # rows = X_test_copy.index.values
    df2 = X_test_copy['DATE'][:n_steps]
    df2 = pd.DataFrame(df2).set_index(X_test.index.values[:-1])
    # df2 = pd.DataFrame(df2).set_index(X_test.index.values[:-1] - 24*FORECASTDAYS+1)

    df = pd.concat([pd.DataFrame(df), df2], axis=0)

    y_all = y_all[:len(y_)+len(y_final)]
    # y_all = y_all[:-24*FORECASTDAYS]

    if plot:
        plt.figure()
        plt.plot(df, y_all, label=f'Real data')
        plt.plot(df2, y_final, label=f'Forecasted', linestyle='--')
        # plt.title(f'{DATASET_NAME} dataset Prediction')
        plt.xlabel('Date', fontsize=FONT_SIZE)
        plt.ylabel('Load [MW]', fontsize=FONT_SIZE)
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE)
        plt.savefig(path+f'/results/{MODE}_noCV_composed_pred_vs_real.pdf')
        plt.show()
        plt.tight_layout()
    r2test = r2_score(y_test, y_final)
    log("The R2 score on the Test set is:\t{:0.4f}".format(r2test))
    n = len(X_test)
    p = X_test.shape[1]
    adjr2_score = 1-((1-r2test)*(n-1)/(n-p-1))
    log("The Adjusted R2 score on the Test set is:\t{:0.4f}".format(
        adjr2_score))

    rmse = np.sqrt(mean_squared_error(y_test, y_final))
    log("RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, y_final)
    log("MAE: %f" % (mae))

    # Fix shape
    if len(y_final) > 1:
        y_final = y_final.ravel()
    if len(y_test) > 1:
        try:
            y_test = y_test.ravel()
        except AttributeError:
            y_test = y_test.values.ravel()

    mape = mean_absolute_percentage_error(y_test, y_final)
    smape = symmetric_mape(y_test, y_final)
    log("MAPE: %.2f%%" % (mape))
    log("sMAPE: %.2f%%" % (smape))

    if plot and True:
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
    if BOXCOX:
        min_y = min(y)
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
        minmax = MinMaxScaler(feature_range=(1, 100))
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
    plt.savefig(path+f'/results/{DATASET_NAME}_{MODE}_feature_importance.pdf')

    # featImportance = pd.concat([pd.DataFrame({'Features':names}),
    #                  pd.DataFrame({'Relative_Importance':importances[indices]})], axis=1, sort=False)


def open_json(model, algorithm, imf):
    filePath = path + \
        f'/src/params/{algorithm}_params_{imf}_{MODE.upper()}.json'
    try:
        # Opening JSON file
        fp = open(filePath)
        local_params = json.load(fp)
    except (FileNotFoundError, OSError, IOError) as e:
        log(f'Hyperparameters JSON file not found: {e}')
        log(f'Use default params...')
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


def saveDecomposedIMFs(y_decomposed_list):
    if not LOAD_DECOMPOSED and (MODE != 'none' or MODE != 'robust-stl'):
        for imf in y_decomposed_list:
            if type(imf) is not type(pd.DataFrame()):
                imf = pd.DataFrame({imf.name: imf.values})
            imf.to_csv(
                path+f'/datasets/{DATASET_NAME}/custom/{MODE}_{imf.columns[0]}_forecast{FORECASTDAYS}_{selectDatasets[0]}-{selectDatasets[-1]}.csv', index=None, header=False)


################
# MAIN PROGRAM
################
# Verify arguments for program execution
for args in sys.argv:
    if args == '-nni':
        enable_nni = True
        plot = False

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

# Split the test data from training/validation data
y_testset = y[(-24*TEST_DAYS):]
X_testset = X[(-24*TEST_DAYS):]
X = X[:(-24*TEST_DAYS)]
y = y[:(-24*TEST_DAYS)]

# Redefine df
global df
df = X['DATE']
# Outlier removal
y = outlierCleaning(y, dataset_name=DATASET_NAME)

if plot and True:
    plt.figure()
    plt.title(f'{DATASET_NAME} dataset demand curve')
    plt.xlabel('Date')
    plt.ylabel('Load [MW]')
    plt.plot(df, y)
    plt.show()
    plt.tight_layout()
    plt.savefig(path+f'/results/{DATASET_NAME}_after_outlierClean.pdf')
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

# Data transformation - BoxCox and MinMaxScaler/StandardScaler
y_transf, lambda_boxcox, sc1, minmax, min_y = data_transformation(y)

# Decompose data
y_decomposed_list = decomposeSeasonal(
    df, y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE)

# Save decomposed data
saveDecomposedIMFs(y_decomposed_list)

# Index for Results
r = 0
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
        X=X, y=y_decomposed, CrossValidation=CROSSVALIDATION, kfold=KFOLD, offset=OFFSET, forecastDays=FORECASTDAYS, dataset_name=DATASET_NAME)
    # Save the current model for further usage
    models.append(model)

    if CROSSVALIDATION:
        decomposePred.append(kfoldPred)
    else:
        decomposePred.append(y_out)
    r += 1

if not enable_nni and not CROSSVALIDATION:
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)
    data_transformation_inverse(
        y_composed, lambda_boxcox, sc1, minmax, min_y, cv=CROSSVALIDATION)

    log("Print and plot the results")
    finalResults.append(Results())
    plotResults(X_=X, y_=y, y_pred=y_composed,
                testSize=testSize, dataset_name=DATASET_NAME)
    finalTest(model=models, X_test=X_testset,
              y_test=y_testset, X_=X, y_=y, testSize=testSize)


if CROSSVALIDATION:
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)
    data_transformation_inverse(
        y_composed, lambda_boxcox, sc1, minmax, min_y, cv=CROSSVALIDATION)
    log("Print and plot the results")
    plotResults(X_=X, y_=y, y_pred=y_composed,
                testSize=testSize, dataset_name=DATASET_NAME)
    finalTest(model=models, X_test=X_testset,
              y_test=y_testset, X_=X, y_=y, testSize=testSize)

if enable_nni:
    log("Publish the results on AutoML nni")
    rmsetestResults = results[0].rmse_per_fold
    rmseScoreAvg = np.mean(rmsetestResults)
    log(f"rmseScoreAvg = {rmseScoreAvg}")
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
