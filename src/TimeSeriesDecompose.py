"""
Time-Series Decomposition
Author: Marcos Yamasaki
04/03/2021
"""
import time
import logging
from log import log
start_time = time.time()
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import holidays
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import xgboost
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats, special
from Results import Results
import sys
from PyEMD import EMD, EEMD, CEEMDAN
# from vmdpy import VMD
import ewtpy
from sklearn import linear_model, cross_decomposition
from sklearn import svm
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, VotingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from skgarden import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
#from RobustSTL import RobustSTL
from sklearn.preprocessing import MinMaxScaler, normalize
import nni
from pandas.plotting import register_matplotlib_converters

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
KFOLD = 40
OFFSET = 0
FORECASTDAYS = 1
NMODES = 6
MODE = 'emd'
BOXCOX = True
MINMAXSCALER = False
DIFF = False
LOAD_DECOMPOSED = False
# Selection of year
#selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
selectDatasets = ["2015","2016","2017","2018"]
# selectDatasets = ["2015"]
# Set algorithm
ALGORITHM = 'ensemble'
# Seasonal component to be analyzed
COMPONENT : str = 'Trend'
###
# Default render
pio.renderers.default = 'browser'
# Default size for plotly export figures
#pio.kaleido.scope.default_width = 1280
#pio.kaleido.scope.default_height = 720
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(14, 6)})
# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(f'\\{os.path.basename(__file__)}','').replace('\\','/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML','')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src','')

log(f"Dataset: {DATASET_NAME}")
log(f"Years: {selectDatasets}")
log(f"CrossValidation: {CROSSVALIDATION}")
log(f"KFOLD: {KFOLD}")
log(f"OFFSET: {OFFSET}")
log(f"FORECASTDAYS: {FORECASTDAYS}")
log(f"NMODES: {NMODES}")
log(f"MODE: {MODE}")
log(f"BOXCOX: {BOXCOX}")
log(f"ALGORITHM: {ALGORITHM}")
log(f"MINMAXSCALER: {MINMAXSCALER}")


# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)


def datasetImport(selectDatasets, dataset_name='ONS'):
    log('Dataset import has been started')
    # Save all files in the folder
    if dataset_name.find('ONS') != -1:
        filename = glob.glob(path + r'/datasets/ONS/*south*.csv')
        filename = filename[0].replace('\\','/')
        dataset = pd.read_csv(filename,index_col=None, header=0, delimiter=";")
        # Select only selected data
        datasetList = []
        for year in selectDatasets:
            datasetList.append(dataset[dataset['DATE'].str.find(year) != -1])
    elif dataset_name.find('ISONewEngland') != -1:
        all_files = glob.glob(path + r'/datasets/ISONewEngland/csv-fixed/*.csv')
        # Initialize dataset list
        datasetList = []
        # Read all csv files and concat them
        for filename in all_files:
            if (filename.find("ISONE") != -1):
                for data in selectDatasets:
                    if (filename.find(data) != -1):
                        df = pd.read_csv(filename,index_col=None, header=0)
                        datasetList.append(df)
    # Concat them all
    dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)
    
    if dataset_name.find('ONS') != -1:
        # replace comma to dot
        dataset['DEMAND'] = dataset['DEMAND'].str.replace(',','.')
        dataset['DATE'] = pd.to_datetime(dataset.DATE, format="%d/%m/%Y %H:%M")
        dataset = dataset.sort_values(by='DATE', ascending=True)
    
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
            X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
        except KeyError:
            X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad'], axis=1)
        # Drop additional unused columns/features
        for columnNames in X.columns:
            if(columnNames.find("5min") != -1):
                X.drop([columnNames], axis=1, inplace=True)
    ## Pre-processing input data 
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
        log(dataset[dataset['DEMAND'].isnull()])
        # Save the NaN indexes
        nanIndex = dataset[dataset['DEMAND'].isnull()].index.values
        # Replace zero values by NaN
        dataset['DEMAND'].replace(0, np.nan, inplace=True)
        #convert to float
        y = dataset['DEMAND'].astype(float)
        y = y.interpolate(method='linear', axis=0).ffill().bfill()
        log(y.iloc[nanIndex])

    # Select Y data
    if dataset_name.find('ONS') != -1:
        y = pd.concat([pd.DataFrame({'DEMAND':y}), dataset['SUBSYSTEM']], axis=1, sort=False)

    return X, y

def featureEngineering(dataset, X, y, selectDatasets, weekday=True, holiday=True, holiday_bridge=False,  dataset_name='ONS'):
    log('Feature engineering has been started')
    # Decouple date and time from dataset
    # Then concat the decoupled date in different columns in X data


    log("Adding date components (year, month, day, holidays and weekdays) to input data")
    # Transform to date type
    X['DATE'] = pd.to_datetime(dataset.DATE)
    
    # log("Use lagged y (demand) to include as input in X")
    # y_lag = pd.DataFrame({f'Demand_lag':y.shift(-FORECASTDAYS*24)})
    # concatlist = [X, y_lag]
    # X = pd.concat(concatlist,axis=1)
    # # Drop null/NaN values    
    # # First save indexes to drop in y
    # drop = X[X[f'Demand_lag'].isnull()].index.values
    # # Drop X
    # X = X.dropna()
    # # Drop y
    # y = y.drop(index=drop)

    date = X['DATE']
    Year = pd.DataFrame({'Year':date.dt.year})
    Month = pd.DataFrame({'Month':date.dt.month})
    Day = pd.DataFrame({'Day':date.dt.day})
    Hour = pd.DataFrame({'HOUR':date.dt.hour})

    if weekday:
        # Add weekday to X data
        Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

    if holiday:
        # Add holidays to X data
        br_holidays = []
        for date2 in holidays.Brazil(years=list(map(int,selectDatasets))).items():
            br_holidays.append(str(date2[0]))

        # Set 1 or 0 for Holiday, when compared between date and br_holidays
        Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in br_holidays else 0 for val in date]})

    
    # Concat all new features into X data
    try: 
        concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
    except (AttributeError, ValueError, KeyError, UnboundLocalError) as e:
        concatlist = [X,Year,Month,Day,Hour]
    X = pd.concat(concatlist,axis=1)

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


    X_all = []
    y_all = []
    
    if holiday_bridge:
        log("Adding bridge days (Mondays / Fridays) to the Holiday column")
        # Holidays on Tuesdays and Thursday may have a bridge day (long weekend)
        # X_tmp = X_all[0][(X_all[0]['Holiday'] > 0).values].drop_duplicates(subset=['Day','Month','Year'])
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
        
            
        Holiday_bridge = pd.DataFrame({'Holiday_bridge':[1 if val in bridgeDayList else 0 for val in date]})


        concatlist = [X,Holiday_bridge]
        X = pd.concat(concatlist,axis=1)

        # Sum the two holidays columns to merge them into one and remove unnecessary columns
        X['Holiday_&_bridge']=X.loc[:,['Holiday','Holiday_bridge']].sum(axis=1)
        X = X.drop(['Holiday','Holiday_bridge'], axis=1)

    if dataset_name.find('ONS') != -1:       
        # Store regions in a list of dataframes
        log('Organize and split input data by different regions')
        unique = X['SUBSYSTEM'].unique()

        for region in unique:
            X_temp = X[X['SUBSYSTEM']==region].reset_index(drop=True)
            X_all.append(X_temp)
            y_temp = y[y['SUBSYSTEM']==region].reset_index(drop=True)
            y_all.append(y_temp)
    
    elif dataset_name.find('ISONewEngland') != -1:
        X_all.append(X)
        y_all.append(y)


    return X_all, y_all


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

def decomposeSeasonal(y_, dataset_name='ONS', Nmodes=3, mode='stl-a'):
    tic = time.time()
    if mode=='stl-a' or mode=='stl-m':
        log('Seasonal and Trend decomposition using Loess (STL) Decomposition has been started')
        data = pd.DataFrame(data=df)

        if dataset_name.find('ONS') != -1:
            try:
                concatlist = [data,pd.DataFrame(y_.drop(['SUBSYSTEM'], axis=1))]
            except (AttributeError, KeyError) as e:
                concatlist = [data,pd.DataFrame(y_)]
        elif dataset_name.find('ISONewEngland') != -1:
            concatlist = [data,pd.DataFrame(y_)]
        data = pd.concat(concatlist,axis=1)

        data.reset_index(inplace=True)
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data.set_index('DATE')
        data = data.drop(['index'], axis=1)
        data.columns = ['DEMAND']
        if mode == 'stl-a':
            model = 'additive'
        elif mode == 'stl-m':
            model = 'multiplicative'
        result = seasonal_decompose(data, period=24, model=model, extrapolate_trend='freq')
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
    elif mode=='emd' or mode=='eemd' or mode=='vmd' or mode=='ceemdan' or mode=='ewt':
        decomposeList = emd_decompose(y_, Nmodes=Nmodes, dataset_name=DATASET_NAME, mode=mode)
    elif mode=='robust-stl':        
        decomposeList = RobustSTL(y_.values.ravel(), 50, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
        labels = ['Observed','Trend','Seasonal','Remainder']
        for i in range(len(decomposeList)):
            decomposeList[i] = pd.DataFrame({labels[i]:decomposeList[i]})

    elif mode=='none':
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
        except AttributeError:
            y_ = y_

    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=25)

    y_pred = clf.fit_predict(pd.DataFrame(y_))
#    outliers_train = y_train.loc[y_pred_train == -1]
    
    negativeOutlierFactor = clf.negative_outlier_factor_
    outliers = y_.loc[negativeOutlierFactor < (negativeOutlierFactor.mean() - negativeOutlierFactor.std()-1)]
    
#    outliers.reindex(list(range(outliers.index.min(),outliers.index.max()+1)),fill_value=0)
    

    outliers_reindex = outliers.reindex(list(range(df.index.min(),df.index.max()+1)))
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
                        yaxis = dict(
                                scaleanchor = "x",
                                scaleratio = 1),
                        xaxis = dict(
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
    y_ = pd.DataFrame(y_).replace([outliers],np.nan)    
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
                        yaxis = dict(
                            scaleanchor = "x",
                            scaleratio = 1),
                        xaxis = dict(
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
    if dataset_name.find('ONS') != -1:
        try:
            X = X.drop(['SUBSYSTEM', 'DATE'], axis=1)
        except KeyError:
            pass 
    elif dataset_name.find('ISONewEngland') != -1:
        try:
            X = X.drop(['DATE'], axis=1)
        except KeyError:
            pass # ignore it
    try:
        if y.columns.str.find("SUBSYSTEM") != -1 and y.columns[0] is not None:
            y = y.drop(['SUBSYSTEM'], axis=1)
        else:
            pass
    except AttributeError:
        pass

    # Shift demand and drop null values
    X, y = get_lagged_y(X, y, forecastDays=forecastDays)
    if len(df) > len(y):
        df = df[:len(y)]

    # Drop unnecessary columns from inputs
    # if y.columns[0].find('IMF_0') != -1:
    #     X = X.drop(['Year','Month','Day','Weekday','Holiday','HOUR','DRYBULB','DEWPNT'], axis=1)
    # elif y.columns[0].find('IMF_1') != -1:
    #     X = X.drop(['Year','HOUR','Month','DRYBULB','Holiday'], axis=1)
    # elif y.columns[0].find('IMF_2') != -1:
    #     X = X.drop(['Year','Holiday','HOUR','DRYBULB'], axis=1)
    # elif y.columns[0].find('IMF_3') != -1:
    #     X = X.drop(['Year','Month','Day','Holiday','Weekday','DEWPNT','DRYBULB'], axis=1)
    # elif y.columns[0].find('IMF_4') != -1:
        # X = X.drop(['Year','Month','Day','Holiday','DRYBULB','DEWPNT','Weekday','HOUR'], axis=1)
    
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
        train_size = round((len(inputs)/kfold) - test_size)
        
        # Offset on Forecast window        
        # offset = test_size*3
        
        if offset > 0:
            log(f'Offset has been set by {offset/24} days')
            # test_size = round((X.shape[0]-offset)/uniqueYears.size/12/2)
            test_size = round(forecastDays*24)
            train_size = round(((len(inputs)-offset)/kfold) - test_size)
    

        train_index = np.arange(0,train_size+offset)
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
                plt.title(f'Electricity Prediction [MW] - with Box-Cox Transformation - {y.columns[0]}')
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
            # regressors.append(('svm', svm.SVR(kernel='rbf', gamma=0.001, C=10000)))
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
            # meta_learner = linear_model.ARDRegression() # 0.88415
            # meta_learner = LinearRegression() # 0.88037
            # meta_learner = linear_model.BayesianRidge() # 0.877
            
            # model = VotingRegressor(estimators=regressors)
            # model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)
            # model = StackingRegressor(estimators=regressors, final_estimator=meta_learner)
            model = GradientBoostingRegressor()
                                        # loss="ls",
                                        # learning_rate=0.0023509843102651725,
                                        # n_estimators=10000,
                                        # subsample=0.0065259491955755415,
                                        # criterion="mse",
                                        # min_samples_split=8,
                                        # min_weight_fraction_leaf=0,
                                        # max_depth=23,
                                        # min_impurity_decrease=0.5,
                                        # max_features="log2",
                                        # alpha=0.9,
                                        # warm_start=True,
                                        # validation_fraction=0.5,
                                        # tol=0.00009659717194630799,
                                        # ccp_alpha=0.7000000000000001,
                                        # random_state=42)
            # if y.columns[0].find('mode_0') != -1:
                # Best configuration so far: gbr; metalearner=ARDR
            # regressors = list()
            # regressors.append(('xgboost', xgboost.XGBRegressor()))
            # regressors.append(('gbr', GradientBoostingRegressor()))
            # regressors.append(('knn', KNeighborsRegressor()))
            # regressors.append(('cart', DecisionTreeRegressor()))
            # regressors.append(('rf', RandomForestRegressor()))
            # regressors.append(('svr', svm.SVR()))
            # regressors.append(('extratrees', ExtraTreesRegressor()))
            # regressors.append(('ridge', linear_model.Ridge()))
            # regressors.append(('pls', cross_decomposition.PLSRegression()))
            # regressors.append(('sgd', linear_model.SGDRegressor()))
            # regressors.append(('bayes'  , linear_model.BayesianRidge()))
            # regressors.append(('lasso', linear_model.LassoLars()))
            # regressors.append(('ard', linear_model.ARDRegression()))
            # regressors.append(('par', linear_model.PassiveAggressiveRegressor()))
            # regressors.append(('theilsen', linear_model.TheilSenRegressor()))
            # regressors.append(('linear', linear_model.LinearRegression()))
            # meta_learner = linear_model.ARDRegression() # 0.88415
            # model = StackingRegressor(estimators=regressors, final_estimator=meta_learner)
            
        #    tscv = TimeSeriesSplit(n_splits=5)

        #     if y.columns[0].find('Trend') != -1:
        #         model = ExtraTreesRegressor()
        #     elif y.columns[0].find('Seasonal') != -1:
        #         model = GradientBoostingRegressor()
        #     elif y.columns[0].find('Residual') != -1:
        #         model = GradientBoostingRegressor()


            model = GradientBoostingRegressor()
            # Choose one model for each IMF
            # if y.columns[0].find('IMF_0') != -1:
            #     model = ExtraTreesRegressor(
            #                                 n_estimators=1000,
            #                                 criterion="mse",
            #                                 max_depth=128,
            #                                 min_samples_split=2,
            #                                 min_samples_leaf=32,
            #                                 min_weight_fraction_leaf=0,
            #                                 max_features="auto",
            #                                 min_impurity_decrease=0,
            #                                 bootstrap=True,
            #                                 warm_start=True,
            #                                 ccp_alpha=0
            #                                 )
            # elif y.columns[0].find('IMF_1') != -1:
            #     model = xgboost.XGBRegressor(
            #                                 # colsample_bytree=0.75,
            #                                 # gamma=0.08,
            #                                 # learning_rate=1,
            #                                 # max_depth=13,
            #                                 # min_child_weight=17,
            #                                 # n_estimators=1600,
            #                                 # reg_alpha=0.05,
            #                                 # reg_lambda=0.31,
            #                                 # subsample=0.9400000000000001,

            #                                 colsample_bytree=0.65,
            #                                 gamma=0.01,
            #                                 learning_rate=0.07,
            #                                 max_depth=21,
            #                                 min_child_weight=5,
            #                                 n_estimators=3200,
            #                                 reg_alpha=0,
            #                                 reg_lambda=0.36,
            #                                 subsample=0.78,
            #                                 seed=42
            #                                 )
            #     model = xgboost.XGBRegressor()
            # elif y.columns[0].find('IMF_2') != -1:
            #     model = GradientBoostingRegressor()
            # elif y.columns[0].find('IMF_3') != -1:
            #     model = GradientBoostingRegressor()
            # elif y.columns[0].find('IMF_4') != -1:
            #     # model = GradientBoostingRegressor(
            #     #                                     loss="ls",
            #     #                                     learning_rate=0.0009633219347502185,
            #     #                                     n_estimators=22000,
            #     #                                     subsample=0.168718268093578,
            #     #                                     criterion="mse",
            #     #                                     min_samples_split=4,
            #     #                                     min_weight_fraction_leaf=0,
            #     #                                     max_depth=28,
            #     #                                     min_impurity_decrease=0,
            #     #                                     max_features="sqrt",
            #     #                                     alpha=0.1,
            #     #                                     warm_start="True",
            #     #                                     validation_fraction=1,
            #     #                                     tol=0.000001810181307798231,
            #     #                                     ccp_alpha=0
            #     #                                 )
            #     model = ExtraTreesQuantileRegressor(
            #                                         n_estimators=1610,
            #                                         criterion="mae",
            #                                         max_depth=256,
            #                                         min_samples_split=10,
            #                                         min_samples_leaf=8,
            #                                         min_weight_fraction_leaf=0,
            #                                         max_features="auto",
            #                                         bootstrap=True,
            #                                         warm_start=False,
            #                                         random_state=42
            #                                         )
            #     model = GradientBoostingRegressor()
            #     # model = AdaBoostRegressor()
            #     # regressors = list()
            #     # regressors.append(('extratreesq',ExtraTreesQuantileRegressor()))
            #     # regressors.append(('gbm',GradientBoostingRegressor()))
            #     # regressors.append(('pls',PLSRegression()))
            #     # regressors.append(('svr',svm.SVR()))
            #     # meta_learner = linear_model.ARDRegression() # 0.88415
            #     # model = StackingRegressor(estimators=regressors, final_estimator=meta_learner)
            # elif y.columns[0].find('IMF_5') != -1:
            #     model = GradientBoostingRegressor(  
            #                                       loss="lad",
            #                                       learning_rate=0.005713111629093579,
            #                                       n_estimators=5250,
            #                                       subsample=0.0013874269369021587,
            #                                       criterion="friedman_mse",
            #                                       min_samples_split=8,
            #                                       min_weight_fraction_leaf=0,
            #                                       max_depth=10,
            #                                       min_impurity_decrease=0.8,
            #                                       max_features="sqrt",
            #                                       alpha=0.9,
            #                                       warm_start=True,
            #                                       validation_fraction=0.30000000000000004,
            #                                       tol=0.000001004846729035755,
            #                                       ccp_alpha=0,
            #                                           random_state=42
            #                                     )
            #     model = GradientBoostingRegressor()
            # elif y.columns[0].find('IMF_6') != -1:
            #     model = GradientBoostingRegressor()
   
        else: # nni enabled
            # model = xgboost.XGBRegressor(
            #                             colsample_bytree=params['colsample_bytree'],
            #                             gamma=params['gamma'],
            #                             learning_rate=params['learning_rate'],
            #                             max_depth=int(params['max_depth']),
            #                             min_child_weight=int(params['min_child_weight']),
            #                             n_estimators=int(params['n_estimators']),
            #                             reg_alpha=params['reg_alpha'],
            #                             reg_lambda=params['reg_lambda'],
            #                             subsample=params['subsample'],
            #                             seed=42)


            # # Choose one model for each IMF
            # if y.columns[0].find('IMF_0') != -1:
            #     model_choosen = params['model']
            # elif y.columns[0].find('IMF_1') != -1:
                
            # elif y.columns[0].find('IMF_2') != -1:
                
            # elif y.columns[0].find('IMF_3') != -1:
               
            # elif y.columns[0].find('IMF_4') != -1:

            # elif y.columns[0].find('IMF_5') != -1:

            # elif y.columns[0].find('IMF_6') != -1:

            # elif y.columns[0].find('IMF_7') != -1:
                
                
            if params['warm_start'] == "True":
                warm_start = True
            elif params['warm_start'] == "False":
                warm_start = False

            if params['min_samples_split'] > 1:
                min_samples_split=int(params['min_samples_split'])
            else:
                min_samples_split=float(params['min_samples_split'])

            model = GradientBoostingRegressor(
                                              loss=params['loss'],
                                              learning_rate=params['learning_rate'],
                                              n_estimators=int(params['n_estimators']),
                                              subsample=params['subsample'],
                                              criterion=params['criterion'],
                                              min_samples_split=min_samples_split,
                                              min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                              max_depth=int(params['max_depth']),
                                              min_impurity_decrease=params['min_impurity_decrease'],
                                              max_features=params['max_features'],
                                              alpha=params['alpha'],
                                              warm_start=warm_start,
                                              validation_fraction=params['validation_fraction'],
                                              tol=params['tol'],
                                              ccp_alpha=params['ccp_alpha'],
                                              random_state=42)
            # if params['min_samples_split'] > 1:
            #     min_samples_split=int(params['min_samples_split'])
            # else:
            #     min_samples_split=float(params['min_samples_split'])
                            
            
            # if params['bootstrap'] == "True":
            #     bootstrap = True
            # else:
            #     bootstrap = False
            # if params['warm_start'] == "True":
            #     warm_start = True
            # elif params['warm_start'] == "False":
            #     warm_start = False
            

            # model = ExtraTreesRegressor(
            #                             n_estimators=int(params['n_estimators']),
            #                             criterion=params['criterion'],
            #                             max_depth=None if params['max_depth']=="None" else params['max_depth'],
            #                             min_samples_split=min_samples_split,
            #                             min_samples_leaf=params['min_samples_leaf'],
            #                             min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            #                             max_features=params['max_features'],
            #                             min_impurity_decrease=params['min_impurity_decrease'],
            #                             bootstrap=bootstrap,
            #                             warm_start=warm_start,
            #                             ccp_alpha=params['ccp_alpha'],
            #                             random_state=42)
            # model = ExtraTreesQuantileRegressor(
            #                             n_estimators=int(params['n_estimators']),
            #                             criterion=params['criterion'],
            #                             max_depth=None if params['max_depth']=="None" else params['max_depth'],
            #                             min_samples_split=min_samples_split,
            #                             min_samples_leaf=params['min_samples_leaf'],
            #                             min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            #                             max_features=params['max_features'],
            #                             bootstrap=bootstrap,
            #                             random_state=42)

        i=0
        
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
            log('------------------------------------------------------------------------')
            log(f'Training for fold {fold_no} ...')
            
            model.fit(X_train, y_train.ravel())
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
                plt.plot(df2, y_pred, label='Predicted Load (fold='+str(i+1)+")")        

            y_pred_train = model.predict(X_train)
            y_pred_train = np.float64(y_pred_train)
            r2train = r2_score(y_train, y_pred_train)
            r2test = r2_score(y_test, y_pred)

            # log("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
            # log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
            n = len(X_test)
            p = X_test.shape[1]
            adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
            # log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            # log("RMSE: %f" % (rmse))

            mae = mean_absolute_error(y_test, y_pred)
            # log("MAE: %f" % (mae))

            try:
                y_test = y_test.values.reshape(y_test.shape[0])
                mape = mean_absolute_percentage_error(y_test, y_pred)
                smape = symmetric_mape(y_test, y_pred)
            except AttributeError:
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
            results[r].rmse_per_fold.append(rmse)
            results[r].mae_per_fold.append(mae)
            results[r].mape_per_fold.append(mape)
            results[r].smape_per_fold.append(smape)
            results[r].name.append(y.columns[0])

            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes            
            # train_index = np.concatenate((train_index, test_index), axis=0)
            train_index = np.arange(train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)
                
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
            # plt.legend()
            plt.show()
            if BOXCOX:
                plt.savefig(path+f'/results/{MODE}_{y.columns[0]}_BoxCox_loadForecast_k-fold_crossvalidation.pdf')
            else:
                plt.savefig(path+f'/results/{MODE}_{y.columns[0]}_legend_loadForecast_k-fold_crossvalidation.pdf')

            # Calculate feature importances
            try:
                importances = model.feature_importances_
                # Sort feature importances in descending order
                indices = np.argsort(importances)[::-1]
                # Rearrange feature names so they match the sorted feature importances
                names = [X.columns[i] for i in indices]
                #plot_feature_importances(importances,Xdata.columns)
                # Create plot
                plt.figure()
                # Create plot title
                plt.title(f"Feature Importance - {y.columns[0]}")
                # Add bars
                plt.bar(range(X.shape[1]), importances[indices])
                # Add feature names as x-axis labels
                plt.xticks(range(X.shape[1]), names, rotation=0)
                # Show plot
                plt.show()
            except:
                pass

        # Print the results: average per fold
        results[r].printResults()

    else: # NOT CROSSVALIDATION
        log(f'Predict only the last {testSize*X.shape[0]/24} days')
        log(f'Prediction on decomposed part: {y.columns[0]}')
        # transform training data & save lambda value
        # y_boxcox, lambda_boxcox = stats.boxcox(y)
        


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)

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
             
        else: # mni enabled
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
            plt.plot(df,y, label = f'Real data - {y.columns[0]}')
            plt.plot(df2,y_pred, label = f'Predicted data - {y.columns[0]}')
            if BOXCOX:
                plt.title(f'{DATASET_NAME} dataset Prediction - with BoxCox')                
                plt.ylabel('Load [MW] - BoxCox')
            else: 
                plt.title(f'{DATASET_NAME} dataset Prediction')
                plt.ylabel('Load [MW]')
            plt.xlabel('Date')
            plt.legend()
            if BOXCOX:
                plt.savefig(path+f'/results/{MODE}_{y.columns[0]}_noCV_BoxCox_pred_vs_real.pdf')
            else:
                plt.savefig(path+f'/results/{MODE}_{y.columns[0]}_noCV_loadForecast_pred_vs_real.pdf')
            plt.show()
        
        y_pred_train = model.predict(X_train)
        r2train = r2_score(y_train, y_pred_train)
        r2test = r2_score(y_test, y_pred)
        log("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
        log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
        log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        log("RMSE: %f" % (rmse))
        
        mae = mean_absolute_error(y_test, y_pred)
        log("MAE: %f" % (mae))
        
        try:
            y_test = y_test.values.reshape(y_test.shape[0])
            mape = mean_absolute_percentage_error(y_test, y_pred)
            smape = symmetric_mape(y_test, y_pred)
        except AttributeError:
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
        # log("\n--- \t{:0.3f} seconds --- Load Forecasting ".format(time.time() - start_time_xgboost)) 

    log("\n--- \t{:0.3f} seconds --- Load Forecasting ".format(time.time() - start_time_xgboost)) 
    return y_pred, testSize, kfoldPred, model
    

def composeSeasonal(decomposePred, model='stl-a'):
    if not CROSSVALIDATION:
        if model == 'stl-a':
            finalPred = sum(decomposePred)
        elif model == 'stl-m':
            finalPred = np.prod(decomposePred)
        elif model=='emd' or model=='eemd' or model=='vmd' or model=='ceemdan' or model=='ewt':
            finalPred = sum(decomposePred)
        elif model=='none':
            finalPred = decomposePred[0]
        elif model=='robust-stl':
            finalPred = decomposePred[1] + decomposePred[2] + decomposePred[3]
    else:
        if model=='none':
            finalPred = decomposePred[0]
        elif model=='robust-stl':
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
            if y_pred.shape[1]==1:
                y_pred = y_pred.reshape(y_pred.shape[0])
            elif y_pred.shape[0]==1:
                y_pred = y_pred.reshape(y_pred.shape[1])
        if dataset_name.find('ONS') != -1:
            try:
                y_ = y_.drop(["SUBSYSTEM"], axis=1)
            except (AttributeError,KeyError) as e:
                pass
        
            
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = testSize, random_state = 0, shuffle = False)
        # Prepare for plotting
        rows = X_test.index
        df2 = df.iloc[rows[0]:]
        
        if plot:
            plt.figure()
            #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
            try:
                plt.plot(df,y_, label = f'Real data - {y_.columns[0]}')
                plt.plot(df2,y_pred, label = f'Predicted data - {y_.columns[0]}')
    #        except AttributeError:
    #            plt.plot(df,y_, label = f'Real data - {y_.name}')
    #            plt.plot(df2,y_pred, label = f'Predicted data - {y_.name}')
            except AttributeError:
                plt.plot(df,y_, label = f'Real data')
                plt.plot(df2,y_pred, label = f'Predicted data')
            plt.title(f'{DATASET_NAME} dataset Prediction')
            plt.xlabel('Date')
            plt.ylabel('Load [MW]')
            plt.legend()
            plt.savefig(path+f'/results/{MODE}_noCV_composed_pred_vs_real.pdf')
            plt.show()
        
        r2test = r2_score(y_test, y_pred)
        log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
        log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        log("RMSE: %f" % (rmse))
        
        mae = mean_absolute_error(y_test, y_pred)
        log("MAE: %f" % (mae))
        
        try:
            y_test = y_test.values.reshape(y_test.shape[0])
            mape = mean_absolute_percentage_error(y_test, y_pred)
            smape = symmetric_mape(y_test, y_pred)
        except AttributeError:
            mape = mean_absolute_percentage_error(y_test, y_pred)
            smape = symmetric_mape(y_test, y_pred)
        log("MAPE: %.2f%%" % (mape))
        log("sMAPE: %.2f%%" % (smape))
        finalResults[0].r2train_per_fold.append(0)
        finalResults[0].r2test_per_fold.append(r2test)
        finalResults[0].rmse_per_fold.append(rmse)
        finalResults[0].mae_per_fold.append(mae)
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
            plt.title(f'{DATASET_NAME} dataset Load Forecasting - Cross-Validation of {KFOLD}-fold')
            plt.xlabel('Date')
            plt.ylabel('Load [MW]')
            plt.plot(df, y_.squeeze(), color='darkgray', label=f'Electricity Demand [MW]')
            
        # Change variable name because of lazyness
        inputs = np.array(X_)
        targets = np.array(y_)

        # Rest fold number
        fold_no = 1
        
        # Forecast X days
        test_size = round(FORECASTDAYS*24)
        train_size = round((len(inputs)/KFOLD) - test_size)
        
        # Offset on Forecast window        
        # offset = test_size*3
        
        if OFFSET > 0:
            log(f'OFFSET has been set by {OFFSET/24} days')
            # test_size = round((X.shape[0]-OFFSET)/uniqueYears.size/12/2)
            test_size = round(FORECASTDAYS*24)
            train_size = round(((len(inputs)-OFFSET)/KFOLD) - test_size)
    

        train_index = np.arange(0,train_size+OFFSET)
        test_index = np.arange(train_size+OFFSET, train_size+test_size+OFFSET)

        for i in range(0, KFOLD):
            finalResults.append(Results())
            X_train = inputs[train_index]
            y_train = targets[train_index]
            try:                
                X_test = inputs[test_index]
                y_test = targets[test_index]
            except IndexError:
                test_index = np.arange(test_index[0],len(inputs))
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
                plt.plot(df2, y_pred[i], label='Predicted Load (fold='+str(i+1)+")")

        
            r2test = r2_score(y_test, y_pred[i])
#            log("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
#            log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
            n = len(X_test)
            p = X_test.shape[1]
            adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
#            log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))

            rmse = np.sqrt(mean_squared_error(y_test, y_pred[i]))
#            log("RMSE: %f" % (rmse))

            mae = mean_absolute_error(y_test, y_pred[i])
#            log("MAE: %f" % (mae))

            try:
                y_test = y_test.values.reshape(y_test.shape[0])
                mape = mean_absolute_percentage_error(y_test, y_pred[i])
                smape = symmetric_mape(y_test, y_pred[i])
            except AttributeError:
                mape = mean_absolute_percentage_error(y_test, y_pred[i])
                smape = symmetric_mape(y_test, y_pred[i])
#            log("MAPE: %.2f%%" % (mape))
#            log("sMAPE: %.2f%%" % (smape))
            
            
            finalResults[0].r2train_per_fold.append(0)
            finalResults[0].r2test_per_fold.append(r2test)
            finalResults[0].rmse_per_fold.append(rmse)
            finalResults[0].mae_per_fold.append(mae)
            finalResults[0].mape_per_fold.append(mape)
            finalResults[0].smape_per_fold.append(smape)
            finalResults[0].name.append(f'kfold_{i}')

            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes            
            # train_index = np.concatenate((train_index, test_index), axis=0)
            train_index = np.arange(train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)
                
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
            plt.savefig(path+f'/results/{MODE}_loadForecast_k-fold_crossvalidation.pdf')


        # Print the results: average per fold
        finalResults[0].printResults()

def test_stationarity(data):
    from statsmodels.tsa.stattools import adfuller
    log('Stationarity test using Augmented Dickey-Fuller unit root test.')
    test_result = adfuller(data.iloc[:,0].values, regression='ct', maxlag=360, autolag='t-stat')
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
            plt.ylim(0,max(2.0/N * np.abs(yf[0:N//2])))
            plt.show()
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
        plt.ylim(0,max(2.0/N * np.abs(yf[0:N//2])))
        plt.show()


def emd_decompose(y_, Nmodes=3, dataset_name='ONS', mode='eemd'):    
    if mode=='emd':
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
        emd.FIXE_H = 8
        emd.nbsym = 6
        emd.spline_kind = 'cubic'
        IMFs = emd.emd(y_series, max_imf=Nmodes)
        return IMFs
    
    def do_eemd():
        if LOAD_DECOMPOSED:
            all_files = glob.glob(path + r'/datasets/ISONewEngland/custom/eemd_IMF*.csv')
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=0)
                    df = df.values.ravel()
                    IMFs.append(df)
        
        else:
            eemd = EEMD(trials=500, noise_width=0.15, DTYPE=np.float16)
            eemd.MAX_ITERATION = 2000
            eemd.noise_seed(42)
            IMFs = eemd(y_series, max_imf=Nmodes)
        return IMFs
    
    def do_vmd():
        #VMD parameters 
        alpha = 2000 #      % moderate bandwidth constraint
        tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
        init = 1        #  % initialize omegas uniformly
        tol = 1e-7 #
        DC = np.mean(y_series)   # no DC part imposed    
        IMFs = VMD(y_series, alpha, tau, Nmodes, DC, init, tol)
        return IMFs

    def do_ceemdan():
        if LOAD_DECOMPOSED:
            all_files = glob.glob(path + r'/datasets/ISONewEngland/custom/ceemdan_IMF*.csv')
            # Initialize dataset list
            IMFs = []
            # Read all csv files and concat them
            for filename in all_files:
                if (filename.find("IMF") != -1) and (filename.find(MODE) != -1):
                    df = pd.read_csv(filename, index_col=None, header=0)
                    df = df.values.ravel()
                    IMFs.append(df)
        # CEEMDAN - Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
        ceemdan = CEEMDAN(trials = 500, epsilon=0.01)
        ceemdan.noise_seed(42)
        IMFs = ceemdan(y_series,max_imf = Nmodes) 
        return IMFs

    def do_ewt():
        # EWT - Empirical Wavelet Transform
        FFTreg = 'average'
        FFTregLen = 200
        gaussSigma = 15
        ewt,_,_ = ewtpy.EWT1D(y_series, N = Nmodes, log = 0,
                        detect = "locmax", 
                        completion = 0, 
                        reg = FFTreg, 
                        lengthFilter = FFTregLen,
                        sigmaFilter = gaussSigma)
        IMFs = []
        for i in range(ewt.shape[1]):
            IMFs.append(ewt[:,i])        
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
        IMFs = do_ceemdan ()
    elif mode == 'ewt':
        IMFs = do_ewt()
    
    toc = time.time()
    log(f"{toc-tic:0.3f} seconds - {printName} has finished.") 
    series_IMFs = []
    for i in range(len(IMFs)):
        series_IMFs.append(pd.DataFrame({f"IMF_{i}":IMFs[i]}))
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
        if xlabel.find('Box') != -1:
            plt.savefig(path+f'/results/{DATASET_NAME}_BoxCox_histogram.pdf')
        else:        
            plt.savefig(path+f'/results/{DATASET_NAME}_demand_histogram.pdf')

def transform_stationary(y_, y_diff=0, invert=False):
    if invert and len(y_diff)>1:
        try:
            result = np.cumsum(np.concatenate([y_.iloc[0],y_diff.ravel()]))                        
        except KeyError:
            result = np.cumsum(np.concatenate([y_.iloc[0],y_diff.values.ravel()]))
        return result[1:]
    elif not invert:    
        return pd.DataFrame(y_).diff()
    
    # Error
    if y_diff==0:
        assert False

def get_lagged_y(X_, y_, forecastDays=FORECASTDAYS):
    log("Use lagged y (demand) to include as input in X")
    label = y_.columns[0]    
    y_lag = y_.shift(-int(forecastDays*24))
    
    try:
        y_lag = y_lag.rename(columns={label:'DEMAND_LAG'})
    except TypeError:
        y_lag = pd.DataFrame({'DEMAND_LAG':y_lag.ravel()})
    concatlist = [X_, y_lag]
    X_ = pd.concat(concatlist,axis=1)
    # Drop null/NaN values    
    # First save indexes to drop in y
    drop = X_[X_['DEMAND_LAG'].isnull()].index.values
    # Drop X
    X_ = X_.dropna()
    # Drop y
    try:
        y_ = y_.drop(index=drop)
    except KeyError:
        pass
    return X_, y_

def finalTest(model, X_test, y_test, X_, y_, testSize):
    log(f"Final test with test data - Forecast {FORECASTDAYS} day(s)")
    if len(df) != len(y_):
        y_ = y_[:len(df)]
    if len(df) != len(X_):
        X_ = X_[:len(df)]

    # Drop subsystem and date columns
    if DATASET_NAME.find('ONS') != -1:
        try:
            X_test = X_test.drop(['SUBSYSTEM', 'DATE'], axis=1)
        except KeyError:
            pass 
    elif DATASET_NAME.find('ISONewEngland') != -1:
        try:
            X_test = X_test.drop(['DATE'], axis=1)
        except KeyError:
            pass # ignore it
    try:
        if y_test.columns.str.find("SUBSYSTEM") != -1 and y_test.columns[0] is not None:
            y_test = y_test.drop(['SUBSYSTEM'], axis=1)
        else:
            pass
    except AttributeError:
        pass

    # Limit the sizing of test set
    # X_test_1 = X_test[:FORECASTDAYS*24].reset_index(drop=True)
    # y_test_1 = y_test[:FORECASTDAYS*24].reset_index(drop=True)    
    # y_lagged = y_[-FORECASTDAYS*24:].reset_index(drop=True)
    
    # Normalize the signal
    if BOXCOX:        
        min_y = min(y_)
        if min_y <= 0:           
            log("Shift negative to positive values + offset 1")
            y_transf = y_+abs(min_y)+1
        else:
            y_transf = y_
        log("Box-Cox transformation")
        if len(y_transf.shape) > 1:
            y_transf = y_transf.reshape(y_transf.shape[0])
        y_transf, lambda_boxcox = stats.boxcox(y_transf)
        y_transf = pd.DataFrame({'DEMAND':y_transf})
        # log("Plot Histogram after Box-Cox Transformation")
        # plot_histogram(y_transf, xlabel='Box-Cox')
    else:
        y_transf = y_
    
    if MINMAXSCALER:            
        label = y_transf.columns[0]
        # sc1 = MinMaxScaler(feature_range=(1,2))
        sc1 = preprocessing.StandardScaler()
        y_transf = sc1.fit_transform(y_transf)
        try:
            y_transf = pd.DataFrame({label:y_transf})
        except ValueError:
            y_transf = pd.DataFrame({label:y_transf.ravel()})
        except AttributeError:
            y_transf = pd.DataFrame({label:y_transf.values.ravel()})
    
    # Decompose the test set signal
    y_decomposed_list = decomposeSeasonal(y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE)
    
    # List of predictions (IMF_0, IMF_1, ...)
    decomposePred = []

    # Forecast the decomposed signals
    for (model, y_decomposed) in zip(models, y_decomposed_list):
        # Get the y lag
        y_lagged = y_decomposed[-FORECASTDAYS*24:].reset_index(drop=True)
        y_lagged = y_lagged.rename(columns={y_lagged.columns[0]:'DEMAND_LAG'})
        
        # Limit the sizing of test set
        X_test_1 = pd.concat([X_test[:FORECASTDAYS*24].reset_index(drop=True), y_lagged], axis=1)
        y_test_1 = y_test[:FORECASTDAYS*24].reset_index(drop=True)
        
        try:
            y_pred = model.predict(X_test_1.values)
        except ValueError:
            y_pred = model.predict(X_test_1)
        decomposePred.append(y_pred)

    # Compose the signal
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)

    # Invert normalization
    if MINMAXSCALER:
        log("Inverse MinMaxScaler transformation")
        try:
            y_composed= sc1.inverse_transform(y_composed)
        except AttributeError:
            y_composed = sc1.inverse_transform(y_composed.reshape(y_out.shape[0],1))
    if BOXCOX:
        log("Inverse Box-Cox transformation")        
        y_composed = special.inv_boxcox(y_composed, lambda_boxcox)                     
        if min_y <= 0: 
            # log("Restore shifted values from positive to negative + offset -1")
            y_composed = y_composed - abs(min_y)-1
    
    
    ### Evaluate results ###
    
    # Change y variable
    y_final = y_composed

    # Split original series into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = testSize, random_state = 0, shuffle = False)
    # Prepare for plotting
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    if plot:
        plt.figure()
        try:
            plt.plot(df,y_, label = f'Real data - {y_.columns[0]}')
            plt.plot(df2,y_final, label = f'Predicted data - {y_.columns[0]}')
        except AttributeError:
            plt.plot(df,y_, label = f'Real data')
            plt.plot(df2,y_final, label = f'Predicted data')
        plt.title(f'{DATASET_NAME} dataset Prediction')
        plt.xlabel('Date')
        plt.ylabel('Load [MW]')
        plt.legend()
        plt.savefig(path+f'/results/{MODE}_noCV_composed_pred_vs_real.pdf')
        plt.show()
    
    r2test = r2_score(y_test, y_final)
    log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
    n = len(X_test)
    p = X_test.shape[1]
    adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
    log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_final))
    log("RMSE: %f" % (rmse))
    
    mae = mean_absolute_error(y_test, y_final)
    log("MAE: %f" % (mae))
    
    try:
        y_test = y_test.values.reshape(y_test.shape[0])
        mape = mean_absolute_percentage_error(y_test, y_final)
        smape = symmetric_mape(y_test, y_final)
    except AttributeError:
        mape = mean_absolute_percentage_error(y_test, y_final)
        smape = symmetric_mape(y_test, y_final)
    log("MAPE: %.2f%%" % (mape))
    log("sMAPE: %.2f%%" % (smape))

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
X_all, y_all = featureEngineering(dataset, X, y, selectDatasets, dataset_name=DATASET_NAME)

# Split the test data from training/validation data
y_testset = y_all[0][-24*60:]
X_testset = X_all[0][-24*60:]
X_all[0] = X_all[0][:-24*60]
y_all[0] = y_all[0][:-24*60]

# Redefine df
df = X_all[0]['DATE']
# Outlier removal
i=0
for y_ in y_all:
    y_all[i] = outlierCleaning(y_, dataset_name=DATASET_NAME)
    i+=1

if plot and True:
    plt.figure()
    plt.title(f'{DATASET_NAME} dataset demand curve')
    plt.xlabel('Date')
    plt.ylabel('Load [MW]')
    plt.plot(df, y_all[0])
    plt.show()
    plt.savefig(path+f'/results/{DATASET_NAME}_after_outlierClean.pdf')
# List of results
results = []
finalResults = []
r = 0 # index

# Initialize fig
fig = go.Figure()
i = 0 # index for y_all

list_IMFs = []
# fast_fourier_transform(y_all)

# Prediction list of different components of decomposition to be assemble in the end
decomposePred = []
listOfDecomposePred = []
models = []


for inputs in X_all:
    log("Plot Histogram")
    plot_histogram(y_all[i], xlabel='Load [MW]')    
    
    
    if BOXCOX:        
        min_y = min(y_all[i])
        if min_y <= 0:           
            log("Shift negative to positive values + offset 1")
            y_transf = y_all[i]+abs(min_y)+1
        else:
            y_transf = y_all[i]
        log("Box-Cox transformation")
        if len(y_transf.shape) > 1:
            y_transf = y_transf.reshape(y_transf.shape[0])
        y_transf, lambda_boxcox = stats.boxcox(y_transf)
        y_transf = pd.DataFrame({'DEMAND':y_transf})
        log("Plot Histogram after Box-Cox Transformation")
        plot_histogram(y_transf, xlabel='Box-Cox')        
    else:
        y_transf = y_all[i]
        try:
            y_transf = pd.DataFrame({'DEMAND':y_transf})
        except ValueError:
            y_transf = pd.DataFrame({'DEMAND':y_transf.ravel()})
    
    if MINMAXSCALER:            
        label = y_transf.columns[0]
        # sc1 = MinMaxScaler(feature_range=(1,2))
        sc1 = preprocessing.StandardScaler()
        y_transf = sc1.fit_transform(y_transf)
        try:
            y_transf = pd.DataFrame({label:y_transf})
        except ValueError:
            y_transf = pd.DataFrame({label:y_transf.ravel()})
        except AttributeError:
            y_transf = pd.DataFrame({label:y_transf.values.ravel()})
    # if DIFF:
    #     log("Differential operation to make time series stationary")
    #     df = df[1:].reset_index(drop=True)
    #     inputs = inputs[1:].reset_index(drop=True)
    #     y_all[i] = y_all[i][1:]
    #     y_transf_save = y_transf
    #     label = y_transf.columns[0]
    #     y_transf = transform_stationary(y_transf)
    #     y_transf = pd.DataFrame({label:y_transf.dropna().values.ravel()})
        
    y_decomposed_list = decomposeSeasonal(y_transf, dataset_name=DATASET_NAME, Nmodes=NMODES, mode=MODE)
    # for y_decomposed2 in y_decomposed_list:
    #     if y_decomposed2.columns[0].find('Observed') != -1:
    #         y_decomposed_list = emd_decompose(y_decomposed2, Nmodes=NMODES, dataset_name=DATASET_NAME)
    #         break
        
    for y_decomposed in y_decomposed_list:
        if type(y_decomposed) is not type(pd.DataFrame()):
            y_decomposed = pd.DataFrame({y_decomposed.name:y_decomposed.values})
        # if y_decomposed.columns[0].find("IMF_4") == -1:
          #   continue
            
        results.append(Results()) # Start new Results instance every loop step
        
        if MINMAXSCALER and False:            
            label = y_decomposed.columns[0]
            sc = MinMaxScaler(feature_range=(-1,1))
            # sc = preprocessing.StandardScaler()
            try:
                y_decomposed = sc.fit_transform(y_decomposed)
            except ValueError:
                y_decomposed = sc.fit_transform(y_decomposed.to_numpy().reshape(y_decomposed.shape[0],1))
#            y_decomposed = sc.fit_transform(y_decomposed)
            try:
                y_decomposed = pd.DataFrame({label:y_decomposed})
            except ValueError:
                y_decomposed = pd.DataFrame({label:y_decomposed.ravel()})
            except AttributeError:
                y_decomposed = pd.DataFrame({label:y_decomposed.values.ravel()})
                
#        test_stationarity(y_decomposed)
        # Load Forecasting
        y_out, testSize, kfoldPred, model = loadForecast(X=inputs, y=y_decomposed, CrossValidation=CROSSVALIDATION, kfold=KFOLD, offset=OFFSET, forecastDays=FORECASTDAYS, dataset_name=DATASET_NAME)        
        # Save the current model for further usage
        models.append(model)

        # if MINMAXSCALER and False:            
        #     y_out = sc.inverse_transform(y_out.reshape(y_out.shape[0],1))
        #     for j in range(len(kfoldPred)):
        #         kfoldPred[j] = sc.inverse_transform(kfoldPred[j].reshape(kfoldPred[j].shape[0],1))
                
        if CROSSVALIDATION:
            decomposePred.append(kfoldPred)
        else:
            decomposePred.append(y_out)
        r+=1
   
    if not enable_nni and not CROSSVALIDATION:        
        log("Join all decomposed y predictions")
        y_composed = composeSeasonal(decomposePred, model=MODE)        
        # if DIFF:
        #     log("Invert differential")
        #     y_composed = transform_stationary(y_transf_save.iloc[-FORECASTDAYS*24:].reset_index(drop=True), y_composed, invert=True)
        if MINMAXSCALER:
            log("Inverse MinMaxScaler transformation")       
            # y_composed[i] = sc1.inverse_transform(y_composed[i])
            y_composed = sc1.inverse_transform(y_composed.reshape(y_out.shape[0],1))
        if BOXCOX:
            log("Inverse Box-Cox transformation")
            y_composed = special.inv_boxcox(y_composed, lambda_boxcox)                     
            if min_y <= 0: 
                log("Restore shifted values from positive to negative + offset -1")
                y_composed = y_composed - abs(min_y)-1
        
        log("Print and plot the results")
        finalResults.append(Results())
        plotResults(X_=inputs, y_=y_all[i], y_pred=y_composed, testSize=testSize, dataset_name=DATASET_NAME)
        finalTest(model=models, X_test=X_testset, y_test=y_testset, X_=inputs, y_=y_all[0], testSize=testSize)

    i+=1 # y_all[i]
       
if CROSSVALIDATION:
    log("Join all decomposed y predictions")
    y_composed = composeSeasonal(decomposePred, model=MODE)
    if MINMAXSCALER:
        log("Inverse MinMaxScaler transformation")
        for i in range(len(y_composed)):            
            # y_composed[i] = sc1.inverse_transform(y_composed[i])
            y_composed[i] = sc1.inverse_transform(y_composed[i].reshape(y_out.shape[0],1))
    if BOXCOX:
        log("Inverse Box-Cox transformation")
        for i in range(len(y_composed)):            
            y_composed[i] = special.inv_boxcox(y_composed[i], lambda_boxcox)                     
            if min_y <= 0: 
                # log("Restore shifted values from positive to negative + offset -1")
                y_composed[i] = y_composed[i] - abs(min_y)-1
    log("Print and plot the results")    
    plotResults(X_=inputs, y_=y_all[0], y_pred=y_composed, testSize=testSize, dataset_name=DATASET_NAME)
    finalTest(model=models, X_test=X_testset, y_test=y_testset, X_=inputs, y_=y_all[0], testSize=testSize)
        
if enable_nni:
    log("Publish the results on AutoML nni")
#    r2testResults = finalResults[0].r2test_per_fold
    r2testResults = results[0].r2test_per_fold
    r2scoreAvg = np.mean(r2testResults)
    log(f"r2test = {r2scoreAvg}")
    nni.report_final_result(r2scoreAvg)    
    # results[0].printResults()
 

log("\n--- \t{:0.3f} seconds --- the end of the file.".format(time.time() - start_time)) 


# trend = pd.concat([df, y_decomposed_list[1]], axis=1)
# seasonal = pd.concat([df, y_decomposed_list[2]], axis=1)
# remainder = pd.concat([df, y_decomposed_list[3]], axis=1)
# trend.to_csv(path+f'/robust-stl_trend_{selectDatasets[0]}.csv', index = None, header=True)
# seasonal.to_csv(path+f'/robust-stl_seasonal_{selectDatasets[0]}.csv', index = None, header=True)
# remainder.to_csv(path+f'/robust-stl_remainder_{selectDatasets[0]}.csv', index = None, header=True)
#
if not LOAD_DECOMPOSED and MODE != 'none':
    for imf in y_decomposed_list:
        if type(imf) is not type(pd.DataFrame()):
            imf = pd.DataFrame({imf.name:imf.values})
        imf.to_csv(path+f'/datasets/{DATASET_NAME}/custom/{MODE}_{imf.columns[0]}_forecast{FORECASTDAYS}_{selectDatasets[0]}-{selectDatasets[-1]}.csv', index=None, header=False)


# Close logging handlers to release the log file
handlers = logging.getLogger().handlers[:]
for handler in handlers:
    handler.close()
    logging.getLogger().removeHandler(handler)