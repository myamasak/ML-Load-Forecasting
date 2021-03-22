"""
Time-Series Decomposition
Author: Marcos Yamasaki
04/03/2021
"""
import time

start_time = time.time()
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import holidays
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, learning_curve, train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import xgboost
import math
from statsmodels.tsa.seasonal import seasonal_decompose

### Constants
# Set True to plot curves 
plot : bool = True
# Seasonal component to be analyzed
COMPONENT : str = 'Residual'
# log file
logname : str = '/TimeSeriesDecompose.log'
###
# Default render
pio.renderers.default = 'browser'
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
# Set path to import dataset and export figures
path = r'%s' % os.getcwd().replace('\\','/')
# Selection of year
selectDatasets = ["2012","2013","2014","2015","2016"]
# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
import logging
logging.basicConfig(filename=path+logname,
                            format='%(asctime)s %(message)s',
                            datefmt='[%m/%d/%Y %H:%M:%S]',
                            filemode='a',
                            level=logging.INFO)

def log(message):
    logging.info(message)
    print(message)

def datasetImport(selectDatasets):
    log('Dataset import has been started')
    # Save all files in the folder
    filename = glob.glob(path + r'/datasets/ONS/*.csv')
    filename = filename[0].replace('\\','/')
    dataset = pd.read_csv(filename,index_col=None, header=0, delimiter=";")

    # Select only selected data
    datasetList = []
    for year in selectDatasets:
        datasetList.append(dataset[dataset['Data'].str.find(year) != -1])
        
    dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)
    
    # replace comma to dot
    dataset['Demanda'] = dataset['Demanda'].str.replace(',','.')

    return dataset

def dataCleaning(dataset):
    log('Data cleaning function has been started')
    # Select X data
    X = dataset.iloc[:, :]
    X = X.drop(['Demanda'], axis=1)

    ## Pre-processing input data 
    # Verify zero values in dataset (X,y)
    log("Any null value in dataset?")
    log(dataset.isnull().any())
    log("How many are they?")
    log(dataset.isnull().sum())
    log("How many zero values?")
    log(dataset.eq(0).sum())
    log("How many zero values in y (Demanda)?")
    log(dataset['Demanda'].eq(0).sum())

    # Set y
    y = dataset['Demanda'].astype(float)

    # Taking care of missing data
    log('Taking care of missing data')
    if (dataset['Demanda'].eq(0).sum() > 0
        or dataset['Demanda'].isnull().any()):    
        log(dataset[dataset['Demanda'].isnull()])
        # Save the NaN indexes
        nanIndex = dataset[dataset['Demanda'].isnull()].index.values
        # Replace zero values by NaN
        dataset['Demanda'].replace(0, np.nan, inplace=True)

        #convert to float
        y = dataset['Demanda'].astype(float)
        
        y = y.interpolate(method='linear', axis=0).ffill().bfill()
        log(y.iloc[nanIndex])


    # Select Y data
    y = pd.concat([pd.DataFrame({'Demanda':y}), dataset['Subsistema']], axis=1, sort=False)

    return X, y

def featureEngineering(dataset, X, selectDatasets):
    log('Feature engineering has been started')
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
    # for s in br_holidays:
    #     if selectDatasets[0] in str(s):
    #         log(s)

    log("Adding holidays and weekdays to input data")
    # Concat all new features into X data
    concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
    X = pd.concat(concatlist,axis=1)

    # Split X data to different subsystems/regions
    Xs = X[X['Subsistema'].str.find("Sul") != -1].reset_index(drop=True)
    Xs = Xs.drop(['Subsistema','Data'],axis=1)

    # Save in Date format
    global df  # set a global variable for easier plot
    df = X[X['Subsistema'].str.find("Sul") != -1]['Data'].reset_index(drop=True)
    

    # Store regions in a list of dataframes
    log('Organize and split input data by different regions')
    unique = X['Subsistema'].unique()
    X_all = []
    y_all = []
    for region in unique:
        X_temp = X[X['Subsistema']==region].reset_index(drop=True)
        X_all.append(X_temp)
        y_temp = y[y['Subsistema']==region].reset_index(drop=True)
        y_all.append(y_temp)

    return X_all, y_all


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def xgboostCalc(X_, y_, CrossValidation=False, kfold=5, offset=0, forecastDays=30, decompose=False, seasonal_component='Trend'):
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

    X = X_.drop(['Subsistema', 'Data'], axis=1)
    y = y_.drop(['Subsistema'], axis=1)
    
    # Define test size by converting days to percentage
    # testSize = 0.05
    testSize = forecastDays*24/X.shape[0]

    if decompose:
        log('Seasonal decomposition has been started')
        data = pd.DataFrame(data=df)
        concatlist = [data,pd.DataFrame(y_.drop(['Subsistema'], axis=1))]
        data = pd.concat(concatlist,axis=1)

        data.reset_index(inplace=True)
        data['Data'] = pd.to_datetime(data['Data'])
        data = data.set_index('Data')
        data = data.drop(['index'], axis=1)
        data.columns = ['Demanda']
        result = seasonal_decompose(data, freq=24, model='additive', extrapolate_trend='freq')
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
        log(f'Seasonal component choosen: {seasonal_component}')
        for component in decomposeList:
            if (seasonal_component == component.columns[0]):
                y = component
                break

    if CrossValidation:
        log('CrossValidation has been started')
        # Change variable name because of lazyness
        inputs = np.array(X)
        targets = np.array(y)

        # Rest fold number
        fold_no = 1
        
        # Forecast X days
        uniqueYears = X['Year'].unique()
        test_size = round((X.shape[0]/uniqueYears.size)/12/2)
        train_size = round((len(inputs)/kfold) - test_size)
        
        # Offset on Forecast window        
        # offset = test_size*3
        
        if offset > 0:
            log(f'Offset has been set by {offset/24} days')
            test_size = round((X.shape[0]-offset)/uniqueYears.size/12/2)
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
            fig.update_layout(title='Load Forecasting',
                                xaxis_title='Date',
                                yaxis_title=f'Demand Prediction [MW] - {y.columns[0]}'
                                )
        
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


            model.fit(X_train, y_train)
            
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

            mape = mean_absolute_percentage_error(y_test, y_pred.reshape(y_pred.shape[0]))
            
            log("MAPE: %.2f%%" % (mape))
            
#            if plot:
#                fig2 = go.Figure()
#                fig2.add_shape(dict(
#                                type="line",
#                                x0=math.floor(min(np.array(y_test))),
#                                y0=math.floor(min(np.array(y_test))),
#                                x1=math.ceil(max(np.array(y_test))),
#                                y1=math.ceil(max(np.array(y_test)))))
#                fig2.update_shapes(dict(xref='x', yref='y'))
#                fig2.add_trace(go.Scatter(x=y_test.reshape(y_test.shape[0]),
#                                        y=y_pred,
#                                        name='Real price VS Predicted Price (fold='+str(i+1)+")",
#                                        mode='markers'))
#                fig2.update_layout(title='Real vs Predicted price',
#                                xaxis_title=f'Real Demand - {y.columns[0]}',
#                                yaxis_title=f'Predicted Load - {y.columns[0]}')
#                fig2.show()
            
            
            # Generate generalization metrics
            # scores = model.evaluate(X_test, y_test, verbose=0)
            
            # log(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores}')
            # acc_per_fold.append(scores * 100)
            # loss_per_fold.append(scores)
            r2train_per_fold.append(r2train)
            r2test_per_fold.append(r2test)
            rmse_per_fold.append(rmse)
            mae_per_fold.append(mae)
            mape_per_fold.append(mape)


            # Increase fold number
            fold_no = fold_no + 1

            # Increase indexes            
            # train_index = np.concatenate((train_index, test_index), axis=0)
            train_index = np.arange(train_index[-1] + 1, train_index[-1] + 1 + train_size + test_size)
                
            test_index = test_index + train_size + test_size

            decomposePred.append(kfoldPred)

        if plot:
            fig.show()
        
        
        # == Provide average scores ==
        log('------------------------------------------------------------------------')
        log('Score per fold')
        for i in range(0, len(r2test_per_fold)):
            log('------------------------------------------------------------------------')
            # log(f'> Fold {i+1} - Loss: {loss_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - r2_score_train: {r2train_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - r2_score_test: {r2test_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - rmse: {rmse_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - mae: {mae_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - mape: {mape_per_fold[i]:.5f}')
        log('------------------------------------------------------------------------')
        log('Average scores for all folds:')
        # log(f'> Loss: {np.mean(loss_per_fold):.5f}')
        log(f'> r2_score_train: {np.mean(r2train_per_fold):.5f} (+- {np.std(r2train_per_fold):.5f})')
        log(f'> r2_score_test: {np.mean(r2test_per_fold):.5f} (+- {np.std(r2test_per_fold):.5f})')
        log(f'> rmse: {np.mean(rmse_per_fold):.5f} (+- {np.std(rmse_per_fold):.5f})')
        log(f'> mae: {np.mean(mae_per_fold):.5f} (+- {np.std(mae_per_fold):.5f})')
        log(f'> mape: {np.mean(mape_per_fold):.5f} (+- {np.std(mape_per_fold):.5f})')
        log('------------------------------------------------------------------------')
        
        listOfDecomposePred.append(decomposePred)


    else:
        log(f'Predict only the last {testSize*X.shape[0]/24} days')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)
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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Save prediction for later sum
        decomposePred.append(y_pred)

            
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
        
        mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred)
        log("MAPE: %.2f%%" % (mape))

        # tscv = TimeSeriesSplit(n_splits=5)
        # scores = cross_val_score(model, X_, y_, cv=tscv, scoring='r2')
        # with np.printoptions(precision=4, suppress=True):
        #     log(scores)
        # log("Loss: {0:.4f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

        # Feature importance of XGBoost
        if plot:
            ax = xgboost.plot_importance(model)
            ax.figure.set_size_inches(11,15)
            ax.figure.savefig(path + f"/results/plot_importance_xgboost_{X_['Subsistema'].unique()[0]}.png")
            ax.figure.show()
        log("\n--- \t{:0.3f} seconds --- XGBoost ".format(time.time() - start_time_xgboost)) 

    if decompose and len(decomposePred) > 2 and not CrossValidation:

        finalPred = decomposePred[0] + decomposePred[1] + decomposePred[2]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = False)    
        rows = X_test.index
        df2 = df.iloc[rows[0]:]
        
        if plot:
            plt.figure()
            #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
            plt.plot(df,y, label = f'Real data - finalPred')
            plt.plot(df2,finalPred, label = f'Predicted data - finalPred')
            plt.title('Prediction - XGBoost')
            plt.legend()
            plt.savefig(path+'/results/pred_vs_real_finalPred.png')
            plt.show()    
        
        r2test = r2_score(y_test, finalPred)
        log("The R2 score on the Test set is:\t{:0.3f}".format(r2test))
        n = len(X_test)
        p = X_test.shape[1]
        adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
        log("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))
        
        rmse = np.sqrt(mean_squared_error(y_test, finalPred))
        log("RMSE: %f" % (rmse))
        
        mae = mean_absolute_error(y_test, finalPred)
        log("MAE: %f" % (mae))
        
        mape = mean_absolute_percentage_error(y_test.to_numpy(), finalPred)
        log("MAPE: %.2f%%" % (mape))



    log("\n--- \t{:0.3f} seconds --- XGBoost ".format(time.time() - start_time_xgboost)) 
    

################
# MAIN PROGRAM
################
# Initial message
log("Time Series Regression - Load forecasting using xgboost algorithm")
# Dataset import 
dataset = datasetImport(selectDatasets)
# Data cleaning and set the input and reference data
X, y = dataCleaning(dataset)
# Include new data 
X_all, y_all = featureEngineering(dataset, X, selectDatasets)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
r2train_per_fold = []
r2test_per_fold = []
rmse_per_fold = []
mae_per_fold = []
mape_per_fold = []

# Initialize fig
fig = go.Figure()
i = 0 # index for y_all

# Prediction list of different components of decomposition to be assemble in the end
decomposePred = []
listOfDecomposePred = []
for inputs in X_all:
    xgboostCalc(X_=inputs, y_=y_all[i], CrossValidation=True , kfold=25, offset=0, forecastDays=14, decompose=True, seasonal_component=COMPONENT)
    i+=1
    break

log("\n--- \t{:0.3f} seconds --- the end of the file.".format(time.time() - start_time)) 


# Close logging handlers to release the log file
handlers = logging.getLogger().handlers[:]
for handler in handlers:
    handler.close()
    logging.getLogger().removeHandler(handler)