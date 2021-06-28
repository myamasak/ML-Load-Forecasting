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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, learning_curve, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import xgboost
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats, special
from Results import Results
import sys
from PyEMD import EMD, EEMD
from vmdpy import VMD
import ewtpy
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, VotingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

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
KFOLD = 60
OFFSET = 365*24
FORECASTDAYS = 15
NMODES = 3
# Set algorithm
ALGORITHM = 'ensemble'
# Seasonal component to be analyzed
COMPONENT : str = 'Trend'
###
# Default render
pio.renderers.default = 'browser'
# Default size for plotly export figures
pio.kaleido.scope.default_width = 1280
pio.kaleido.scope.default_height = 720
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(f'\\{os.path.basename(__file__)}','').replace('\\','/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML','')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src','')

# Selection of year
#selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
selectDatasets = ["2015","2016","2017","2018"]
# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)


def datasetImport(selectDatasets, dataset_name='ONS'):
    log('Dataset import has been started')
    # Save all files in the folder
    if dataset_name.find('ONS') != -1:
        filename = glob.glob(path + r'/datasets/ONS/*allregions*.csv')
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
    
#    dataset = dataset.iloc[:-24*60,:]
    return dataset

def dataCleaning(dataset, dataset_name='ONS'):
    log('Data cleaning function has been started')
    # Select X data
    X = dataset.iloc[:, :]
    if dataset_name.find('ONS') != -1:
        X = X.drop(['DEMAND'], axis=1)
    elif dataset_name.find('ISONewEngland') != -1:
        X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
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

def featureEngineering(dataset, X, selectDatasets, holiday_bridge=False, dataset_name='ONS'):
    log('Feature engineering has been started')
    # Decouple date and time from dataset
    # Then concat the decoupled date in different columns in X data


    log("Adding date components (year, month, day, holidays and weekdays) to input data")
    # Transform to date type
    X['DATE'] = pd.to_datetime(dataset.DATE)
    # X['DATE'] = pd.to_datetime(dataset.DATE, format="%d/%m/%Y %H:%M")

    date = X['DATE']
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

    # Set 1 or 0 for Holiday, when compared between date and br_holidays
    Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in br_holidays else 0 for val in date]})

    # Concat all new features into X data
    concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
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

def decomposeSeasonal(y_, dataset_name='ONS'):
    log('Seasonal decomposition has been started')
    data = pd.DataFrame(data=df)

    if dataset_name.find('ONS') != -1:
        try:
            concatlist = [data,pd.DataFrame(y_.drop(['SUBSYSTEM'], axis=1))]
        except AttributeError:
            concatlist = [data,pd.DataFrame(y_)]
    elif dataset_name.find('ISONewEngland') != -1:
        concatlist = [data,pd.DataFrame(y_)]
    data = pd.concat(concatlist,axis=1)

    data.reset_index(inplace=True)
    data['DATE'] = pd.to_datetime(data['DATE'])
    data = data.set_index('DATE')
    data = data.drop(['index'], axis=1)
    data.columns = ['DEMAND']
    result = seasonal_decompose(data, period=24, model='additive', extrapolate_trend='freq')
    result.trend.reset_index(drop=True, inplace=True)
    result.seasonal.reset_index(drop=True, inplace=True)
    result.resid.reset_index(drop=True, inplace=True)
    result.observed.reset_index(drop=True, inplace=True)
    result.trend.columns = ['Trend']
    result.seasonal.columns = ['Seasonal']
    result.resid.columns = ['Residual']
    result.observed.columns = ['Observed']
    decomposeList = [result.trend, result.seasonal, result.resid, result.observed]

    # Select one component for seasonal decompose
    # REMOVE FOR NOW
    # log(f'Seasonal component choosen: {seasonal_component}')
    # for component in decomposeList:
    #     if (seasonal_component == component.columns[0]):
    #         y = component
    #         break

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
        fig.write_image(f"{DATASET_NAME}_outliers_"+columnName+".pdf")
    
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

def xgboostCalc(X_, y_, CrossValidation=False, kfold=5, offset=0, forecastDays=15, dataset_name='ONS'):
    log("XGBoost algorithm has been started")
    start_time_xgboost = time.time()
    
    # from sklearn.preprocessing import MinMaxScaler
    # sc = StandardScaler()
    # sc = MinMaxScaler()
    # X_ = sc.fit_transform(X_)
    
    global df, fig
    # Plot 
    if plot:
        fig = go.Figure()
    
    # Drop subsystem and date columns
    if dataset_name.find('ONS') != -1:
        try:
            X = X_.drop(['SUBSYSTEM', 'DATE'], axis=1)
        except KeyError:
            pass 
    elif dataset_name.find('ISONewEngland') != -1:
        try:
            X = X_.drop(['DATE'], axis=1)
        except KeyError:
            pass # ignore it
    try:
        if y_.columns.str.find("SUBSYSTEM") != -1:
            y = y_.drop(['SUBSYSTEM'], axis=1)
        else:
            y = y_
    except AttributeError:
        y = y_
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
        uniqueYears = X['Year'].unique()
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
            fig.add_trace(go.Scatter(x=df,
                                        y=y.squeeze(),
                                        name=f'Electricity Demand [MW] - {y.columns[0]}',
                                        mode='lines'))
            # Edit the layout            
            fig.update_layout(title=f'{dataset_name} dataset Load Forecasting - Cross-Validation of {kfold}-fold',
                                xaxis_title='DATE',
                                yaxis_title=f'Demand Prediction [MW] - {y.columns[0]}'
                                )
        
        if not enable_nni:
#            model = xgboost.XGBRegressor()
#                                        colsample_bytree=0.8,
#                                        gamma=0.3,
#                                        learning_rate=0.03,
#                                        max_depth=7,
#                                        min_child_weight=6.0,
#                                        n_estimators=1000,
#                                        reg_alpha=0.75,
#                                        reg_lambda=0.01,
#                                        subsample=0.95,
#                                        seed=42)
            # Best configuration so far: knn, cart, rf, gbr; metalearner=ARDR
            regressors = list()
#            regressors.append(('xgboost', xgboost.XGBRegressor()))
#            regressors.append(('knn', KNeighborsRegressor()))
#            regressors.append(('cart', DecisionTreeRegressor()))
#            regressors.append(('rf', RandomForestRegressor()))
#            regressors.append(('rf', RandomForestRegressor(n_estimators=750,
#                                                           max_depth=32,
#                                                           min_samples_split=2,
#                                                           min_samples_leaf=1,
#                                                           max_features="auto",
#                                                           max_leaf_nodes=None,
#                                                           min_impurity_decrease=0.001,
#                                                           bootstrap=True,
#                                                           random_state=42,
#                                                           n_jobs=-1)))
#            regressors.append(('svm', svm.SVR(kernel='rbf', gamma=0.001, C=10000)))
            regressors.append(('gbr', GradientBoostingRegressor()))
#                                      n_estimators=750,
#                                      learning_rate=0.1,
#                                      max_depth=3,
#                                      random_state=42)))
#            regressors.append(('extratrees', ExtraTreesRegressor()))
            # regressors.append(('sgd', linear_model.SGDRegressor()))
#            regressors.append(('bayes'  , linear_model.BayesianRidge()))
#            regressors.append(('lasso', linear_model.LassoLars()))
#            regressors.append(('ard', linear_model.ARDRegression()))
#            regressors.append(('par', linear_model.PassiveAggressiveRegressor()))
#            regressors.append(('theilsen', linear_model.TheilSenRegressor()))
#            regressors.append(('linear', linear_model.LinearRegression()))
            
            # define meta learner model
#            meta_learner = GradientBoostingRegressor() # 0.85873
#            meta_learner = ExtraTreesRegressor() # 0.85938
#            meta_learner = linear_model.TheilSenRegressor() # 0.87946      
            meta_learner = linear_model.ARDRegression() # 0.88415
#            meta_learner = LinearRegression() # 0.88037
#            meta_learner = linear_model.BayesianRidge() # 0.877
        
#            model = VotingRegressor(estimators=regressors)
#            model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)
            model = StackingRegressor(estimators=regressors, n_jobs=-1, final_estimator=meta_learner)

   
        else:
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

        i=0
        kfoldPred = []
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
            
            # Predict using test data
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
                fig.add_trace(go.Scatter(x=df2,
                                        y=y_pred,
                                        name='Predicted Load (fold='+str(i+1)+")",
                                        mode='lines'))

        

            y_pred_train = model.predict(X_train)
            y_pred_train = np.float64(y_pred_train)

            r2train = r2_score(y_train, y_pred_train)
            r2test = r2_score(y_test, y_pred)
#            log("The R2 score on the Train set is:\t{:0.3f}".format(r2train))
#            log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
            n = len(X_test)
            p = X_test.shape[1]
            adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
#            log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#            log("RMSE: %f" % (rmse))

            mae = mean_absolute_error(y_test, y_pred)
#            log("MAE: %f" % (mae))

            try:
                y_test = y_test.values.reshape(y_test.shape[0])
                mape = mean_absolute_percentage_error(y_test, y_pred)
                smape = symmetric_mape(y_test, y_pred)
            except AttributeError:
                mape = mean_absolute_percentage_error(y_test, y_pred)
                smape = symmetric_mape(y_test, y_pred)
#            log("MAPE: %.2f%%" % (mape))
#            log("sMAPE: %.2f%%" % (smape))
            
        #    if plot:
        #        fig2 = go.Figure()
        #        fig2.add_shape(dict(
        #                        type="line",
        #                        x0=math.floor(min(np.array(y_test))),
        #                        y0=math.floor(min(np.array(y_test))),
        #                        x1=math.ceil(max(np.array(y_test))),
        #                        y1=math.ceil(max(np.array(y_test)))))
        #        fig2.update_shapes(dict(xref='x', yref='y'))
        #        fig2.add_trace(go.Scatter(x=y_test.reshape(y_test.shape[0]),
        #                                y=y_pred,
        #                                name='Real price VS Predicted Price (fold='+str(i+1)+")",
        #                                mode='markers'))
        #        fig2.update_layout(title='Real vs Predicted price',
        #                        xaxis_title=f'Real Demand - {y.columns[0]}',
        #                        yaxis_title=f'Predicted Load - {y.columns[0]}')
        #        fig2.show()
            
            
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
            fig.update_layout(
                font=dict(size=12),
                legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(
                size=12)
            ))
            fig.show()
            fig.write_image(file=path+'/results/xgboost_k-fold_crossvalidation.svg', width=921, height=618)

            # Print the results: average per fold
            results[r].printResults()

    else:
        log(f'Predict only the last {testSize*X.shape[0]/24} days')
        log(f'Prediction on decomposed part: {y.columns[0]}')
        # transform training data & save lambda value
        # y_boxcox, lambda_boxcox = stats.boxcox(y)
        


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)

        if not enable_nni:
            model = xgboost.XGBRegressor(
                                         colsample_bytree=0.8,
                                         gamma=0.3,
                                         learning_rate=0.03,
                                         max_depth=7,
                                         min_child_weight=6.0,
                                         n_estimators=1000,
                                         reg_alpha=0.75,
                                         reg_lambda=0.01,
                                         subsample=0.95,
                                         seed=42)

#            from tensorflow.keras.models import Sequential
#            from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten, TimeDistributed
#            from tensorflow.keras.callbacks import EarlyStopping

#            ann_model = Sequential()
#            ann_model.add(Dense(16))
#            ann_model.add(LeakyReLU(alpha=0.05))            
#            # Adding the hidden layers
#            for i in range(8):
#                ann_model.add(Dense(units = 16))
#                ann_model.add(LeakyReLU(alpha=0.05))
#            # output layer
#            ann_model.add(Dense(units = 1))
#            # Compiling the ANN
#            model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
#            early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

            regressors = list()
            regressors.append(('xgboost', xgboost.XGBRegressor()))
            regressors.append(('knn', KNeighborsRegressor()))
            regressors.append(('cart', DecisionTreeRegressor()))
            regressors.append(('rf', RandomForestRegressor()))
            regressors.append(('svm', svm.SVR()))
            # regressors.append(('sgd', linear_model.SGDRegressor()))
            # regressors.append(('bayes'  , linear_model.BayesianRidge()))
            # regressors.append(('lasso', linear_model.LassoLars()))
            # regressors.append(('ard', linear_model.ARDRegression()))
            # regressors.append(('par', linear_model.PassiveAggressiveRegressor()))
            # regressors.append(('theilsen', linear_model.TheilSenRegressor()))
            # regressors.append(('linear', linear_model.LinearRegression()))
            
            # define meta learner model
#            meta_learner = xgboost.XGBRegressor()
            meta_learner = LinearRegression()
            
#            model.add(regressors)
#            model.add_meta(meta_learner)
#            model = StackingRegressor(estimators=regressors, final_estimator=meta_learner)
            model = VotingRegressor(estimators=regressors, n_jobs=-1, verbose=True)

                # svm.SVR(kernel='poly',C=1)]      
                # linear_model.SGDRegressor(),
                # linear_model.BayesianRidge(),
                # linear_model.LassoLars(),
                # linear_model.ARDRegression(),
                # linear_model.PassiveAggressiveRegressor(),
                # linear_model.TheilSenRegressor(),
                # linear_model.LinearRegression()]
   
        else:
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
#        for model in regressors:
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        # y_pred = special.inv_boxcox(y_pred, lambda_boxcox)
        # Prepare for plotting
        rows = X_test.index
        df2 = df.iloc[rows[0]:]
        
        if plot:
            plt.figure()
            #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
            plt.plot(df,y, label = f'Real data - {y.columns[0]}')
            plt.plot(df2,y_pred, label = f'Predicted data - {y.columns[0]}')
            plt.title('Prediction - XGBoost')
            plt.legend()
            plt.savefig(path+'/results/pred_vs_real.png')
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
        log("\n--- \t{:0.3f} seconds --- XGBoost ".format(time.time() - start_time_xgboost)) 

    return y_pred, testSize
    log("\n--- \t{:0.3f} seconds --- XGBoost ".format(time.time() - start_time_xgboost)) 
    

def composeSeasonal(decomposePred, model='additive'):
    if model == 'additive':
        finalPred = decomposePred[0] + decomposePred[1] + decomposePred[2]
    elif model == 'multiplicative':
        finalPred = decomposePred[0] * decomposePred[1] * decomposePred[2]
    return finalPred


def plotResults(X_, y_, y_pred, testSize, dataset_name='ONS'):
    if dataset_name.find('ONS') != -1:
        y_ = y_.drop(["SUBSYSTEM"], axis=1)
        
    # LabelEncoder + BoxCox
#    le_y = preprocessing.LabelEncoder()
#    y_ = le_y.fit_transform(y_.values.ravel())
#    y_, lambda_boxcox = stats.boxcox(y_+1)
#    y_ = pd.DataFrame({'DEMAND':y_})
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
        except AttributeError:
            plt.plot(df,y_, label = f'Real data - {y_.name}')
            plt.plot(df2,y_pred, label = f'Predicted data - {y_.name}')
        plt.title('Prediction - XGBoost')
        plt.legend()
        plt.savefig(path+'/results/pred_vs_real.png')
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
    
    mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred)
    log("MAPE: %.2f%%" % (mape))

    smape = symmetric_mape(y_test.to_numpy(), y_pred)
    log("sMAPE: %.2f%%" % (smape))

def test_stationarity(data):
    from statsmodels.tsa.stattools import adfuller
    test_result = adfuller(data.iloc[:,0].values, regression='ct')
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


def emd_decompose(y_, Nmodes=5, dataset_name='ONS'):
    log("Empirical Mode Decomposition (EMD) has been started")
    def do_emd():
        #% EMD features
        tic = time.time()
        emd = EMD()
        emd.MAX_ITERATION = 2000
        IMFs = emd.emd(y_series, max_imf=Nmodes)
        toc = time.time()        
        return IMFs

    if DATASET_NAME.find("ONS") != -1:
        y_series = np.array(y_)
        if y_series.shape[1] == 1:
            y_series = y_series.reshape(y_series.shape[0])
        elif y_series.shape[0] == 1:
            y_series = y_series.reshape(y_series.shape[1])
            IMFs = do_emd()
    else:
        y_series = np.array(y_)
        if y_series.shape[0] == 1:
            y_series = y_series.reshape(y_series.shape[1])
        elif y_series.shape[1] == 1:
            y_series = y_series.reshape(y_series.shape[0])
        IMFs = do_emd()

    series_IMFs = []
    for i in range(len(IMFs)):
        series_IMFs.append(pd.DataFrame({f"mode_{i}":IMFs[i]}))
    return series_IMFs

def get_training_set_for_same_period(X_train, y_train, X_test, y_test, forecastDays=15, dataset_name='ONS'):
    log("Searching the same period of prediction series...")
    X_train[X_train['Month'] == X_test['Month']]
    X_train[X_train['Day'] == X_test['Day']]
    X_train[X_train['Hour'] == X_test['Hour']]

def plot_histogram(y_):
    if plot and False:
        plt.figure()
        sns.histplot(y_)
        plt.ylabel("Occurrences")
        plt.legend()
    #    plt.figure()
    #    sns.kdeplot(y_.values.ravel())
    #    plt.legend()
    
################
# MAIN PROGRAM
################
# Verify arguments for program execution
for args in sys.argv:
    if args == '-nni':
        enable_nni = True
        plot = False
import nni
params = nni.get_next_parameter()     
# Initial message
log("Time Series Regression - Load forecasting using xgboost and other algorithms")
# Dataset import 
dataset = datasetImport(selectDatasets, dataset_name=DATASET_NAME)
# Data cleaning and set the input and reference data
X, y = dataCleaning(dataset, dataset_name=DATASET_NAME)
# Include new data 
X_all, y_all = featureEngineering(dataset, X, selectDatasets, dataset_name=DATASET_NAME)
# Outlier removal
i=0
for y_ in y_all:
    y_all[i] = outlierCleaning(y_, dataset_name=DATASET_NAME)
    i+=1

# List of results
results = []
r = 0 # index

# Initialize fig
fig = go.Figure()
i = 0 # index for y_all

list_IMFs = []
# fast_fourier_transform(y_all)


#y_decoded = le.inverse_transform(encoded)

# Prediction list of different components of decomposition to be assemble in the end
decomposePred = []
listOfDecomposePred = []
for inputs in X_all:
   y_decomposed_list = decomposeSeasonal(y_all[i], dataset_name=DATASET_NAME)   
#   for y_decomposed2 in y_decomposed_list:
#       if y_decomposed2.columns[0].find('Observed') != -1:
#           y_decomposed_list = emd_decompose(y_decomposed2, Nmodes=NMODES, dataset_name=DATASET_NAME)
#           break
   for y_decomposed in y_decomposed_list:
       results.append(Results()) # Start new Results instance every loop step

       if y_decomposed.columns[0].find("Observed") == -1:           
           continue
#       le_X = preprocessing.LabelEncoder()
#       inputs = le_X.fit_transform(inputs)
       # transform training data & save lambda value
#       if y_decomposed.columns[0].find("Residual") != -1:
       # Label encoder for only positive numbers
#       le_y = preprocessing.LabelEncoder()
#       y_decomposed = le_y.fit_transform(y_decomposed.values.ravel())
#       save_y = y_decomposed - le_y.inverse_transform(y_decomposed)
#       plot_histogram(y_decomposed)
       # Box-Cox transformation + shift 1000 integer
       y_decomposed, lambda_boxcox = stats.boxcox(y_decomposed.values.ravel())
       y_decomposed = pd.DataFrame({'DEMAND':y_decomposed})
#       plot_histogram(y_decomposed)
       y_out, testSize = xgboostCalc(X_=inputs, y_=y_decomposed, CrossValidation=CROSSVALIDATION, kfold=KFOLD, offset=OFFSET, forecastDays=FORECASTDAYS, dataset_name=DATASET_NAME)        
       # Inverse Box-Cox transformation
#       y_out = special.inv_boxcox(y_out, lambda_boxcox)
       # Shift -1000
#       y_out = y_out-1000
       # Remove label encoder
#       y_out = y_out - save_y[-len(y_out):]
#       y_out = le_y.inverse_transform(y_out)
       decomposePred.append(y_out)
       r+=1       
       if enable_nni:
           break # stop on trend component
   
   if not enable_nni:
       if not CROSSVALIDATION:
           # Join all decomposed y predictions
           y_composed = composeSeasonal(decomposePred)
           # Print and plot the results
           plotResults(X_=inputs, y_=y_all[i], y_pred=y_composed, testSize=testSize, dataset_name=DATASET_NAME)
       i+=1
       
# Publish the results on AutoML nni
if enable_nni:
   if COMPONENT.find('Trend') != -1:
       r2testResults = results[0].r2test_per_fold # Only for one seasonal component - Trend
   else:
       r2testResults = results[3].r2test_per_fold # Observed component
   r2scoreAvg = np.mean(r2testResults)
   nni.report_final_result(r2scoreAvg)
 

log("\n--- \t{:0.3f} seconds --- the end of the file.".format(time.time() - start_time)) 


# Close logging handlers to release the log file
handlers = logging.getLogger().handlers[:]
for handler in handlers:
    handler.close()
    logging.getLogger().removeHandler(handler)