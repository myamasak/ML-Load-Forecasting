# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import time
start_time = time.time()
import numpy as np
import pandas as pd

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt
import calendar
import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV   #Perforing grid search




# Print configs
pd.options.display.max_columns = None
pd.options.display.width=1000




# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
    

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML-Load-Forecasting'

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONewEngland/csv-fixed/*.csv') + \
            glob.glob(path + r'/datasets/ISONewEngland/holidays/*.csv')

# Select datasets 
#selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
#selectDatasets = ["2015","2016","2017","2018","2019"]

# Initialize dataset list
datasetList = []
#holidayList = []


# Read all csv files and concat them
for filename in all_files:
    if (filename.find("ISONE") != -1):
        for data in selectDatasets:
            if (filename.find(data) != -1):
                df = pd.read_csv(filename,index_col=None, header=0)
                datasetList.append(df)
#    if (filename.find("holidays") != -1):
#        for data in selectDatasets:
#            if (filename.find(data) != -1):
#                df = pd.read_csv(filename,index_col=None, header=0, sep=';', error_bad_lines=False)
#                holidayList.append(df)

# Concat
dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)
#holidays = pd.concat(holidayList, axis=0, sort=False, ignore_index=True)

# Pre-processing holidays data
#calendar.day_name[datetime.datetime.today().weekday()]
#The day of the week with Monday=0, Sunday=6.
#days = dict(zip(calendar.day_name, range(7)))
#weekdayList = []
#for weekday in holidays['Weekday']:
#     weekdayList.append(days[weekday])

# Add weekday number
#holidays['Weekday_number'] = weekdayList

# Drop duplicated holiday dates
#holidays.drop_duplicates(subset=['Date'], keep=False, inplace=True)
#holidays.reset_index(drop=True,inplace=True)




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
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)


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
    from sklearn.preprocessing import Imputer
    y_matrix = y.as_matrix()
    y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer = imputer.fit(y_matrix)
    y =  imputer.transform(y_matrix)


# Decouple date and time from dataset
# Then concat the decoupled date in different columns in X data
#date = pd.DataFrame() 
date = pd.to_datetime(dataset.Date)
dataset['Date'] = pd.to_datetime(dataset.Date)

date = dataset.Date
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'Hour':dataset.Hour})

# Add weekday to X data
Weekday = pd.DataFrame({'Weekday':date.dt.dayofweek})

# Add holidays to X data
us_holidays = []
for date2 in holidays.UnitedStates(years=[2009,2010,2011,2012,2013,2014,2015,2016,2017]).items():
    us_holidays.append(str(date2[0]))

Holiday = pd.DataFrame({'Holiday':[1 if str(val).split()[0] in us_holidays else 0 for val in date]})




# Define season given a timestamp
#Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
#seasons_us = [('Winter', (dt.date(Y,  1,  1),  dt.date(Y,  3, 20))),
#           ('Spring', (dt.date(Y,  3, 21),  dt.date(Y,  6, 20))),
#           ('Summer', (dt.date(Y,  6, 21),  dt.date(Y,  9, 22))),
#           ('Autumn', (dt.date(Y,  9, 23),  dt.date(Y, 12, 20))),
#           ('Winter', (dt.date(Y, 12, 21),  dt.date(Y, 12, 31)))]
#
#def get_season(now):
#    if isinstance(now, dt.datetime):
#        now = now.date()
#    now = now.replace(year=Y)
#    return next(season for season, (start, end) in seasons_us
#                if start <= now <= end)
#
##print(get_season(date.today()))
#
#dateList = []
#for repDate in date:
#    dateList.append(get_season(repDate))
#
#Season = pd.DataFrame({'Season':dateList})
#
#from sklearn.preprocessing import LabelEncoder
## creating initial dataframe
#season_types = ('Winter','Spring','Summer','Autumn')
#season_df = pd.DataFrame(season_types, columns=['Season'])
## creating instance of labelencoder
#labelencoder = LabelEncoder()
## Assigning numerical values and storing in another column
#Season['Season'] = labelencoder.fit_transform(Season['Season'])


# Concat all new features into X data
concatlist = [X,Year,Month,Day,Weekday,Hour,Holiday]
X = pd.concat(concatlist,axis=1)

# Set df to x axis to be plot
df = dataset['Date']


# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)





# Splitting the dataset into the Training set and Test set
# Forecast 30 days - Calculate testSize in percentage
#testSize = (30*24)/(y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)

# Plot actual data
plt.figure(1)
plt.plot(df,y, color = 'gray', label = 'Real data')
plt.legend()
plt.ion()
plt.show()


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def seasonDecomposeCalc():
    start_time_seasonDecompose = time.time()
    
    #from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
    #import statsmodels.api as sm
    data = pd.DataFrame(data=df)
    concatlist = [data,pd.DataFrame(y)]
    data = pd.concat(concatlist,axis=1)
    
    data.reset_index(drop=True,inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    
    # data.sort_values('DEMAND', ascending=False).head(10)
    
    data[data.isnull().any(axis=1)]
    
    
    # result = seasonal_decompose(data.round(0), model='multiplicative', freq=24, extrapolate_trend='freq')
    result = seasonal_decompose(data, model='multiplicative')
#    result = sm.tsa.seasonal_decompose(data)
    result.plot()
    plt.show
    
    print("\n--- \t{:0.3f} seconds --- Seasonal Decompose".format(time.time() - start_time_seasonDecompose))

def featImportanceCalc():
    
    start_time_featImportance = time.time()
    
    ## Feature importance
    # Import random forest
    #from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor  
    
    # Create decision tree classifer object
    #clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf = RandomForestRegressor(random_state=0, n_jobs=-1)
    
    Xdata = dataset.iloc[:, :]
    Xdata = Xdata.drop(['Date','Hour','DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','SYSLoad'], axis=1)
    concatlist = [Xdata,Year,Month,Day,Hour]
    Xdata = pd.concat(concatlist,axis=1)
    
    # Replace NaN values by 0
    Xdata.replace(np.nan, 0, inplace= True)
        
    # Train model
    model = clf.fit(Xdata, y)
    
    # Calculate feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [Xdata.columns[i] for i in indices]
    
    #plot_feature_importances(importances,Xdata.columns)
    
    # Create plot
    plt.figure()
    
    # Create plot title
    plt.title("Feature Importance")
    
    # Add bars
    plt.bar(range(Xdata.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(Xdata.shape[1]), names, rotation=0)
    
    # Show plot
    plt.show()
    
    print("\n--- \t{:0.3f} seconds --- Feature Importance".format(time.time() - start_time_featImportance))

def decisionTreeCalc():
    start_time_decisionTree = time.time()

    # import the regressor 
    from sklearn.tree import DecisionTreeRegressor 
    
    # create a regressor object 
    model = DecisionTreeRegressor(random_state = 0) 
    
    # fit the regressor with X and Y data 
    #model.fit(X, y) 
    model.fit(X_trainsc, y_train)
    
    y_pred = model.predict(X_testsc)
    
    
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - Decision Tree')
    plt.legend()
    plt.show()
    
    
    y_pred_train = model.predict(X_trainsc)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    
    print("\n--- \t{:0.3f} seconds --- Decision Tree".format(time.time() - start_time_decisionTree))


    # import export_graphviz 
    #from sklearn.tree import export_graphviz  
      
    # export the decision tree to a tree.dot file 
    # for visualizing the plot easily anywhere 
    #export_graphviz(model, out_file ='tree.dot', 
    #               feature_names = X.columns.values.tolist()) 
    #

def randForestCalc():
    start_time_randForest = time.time()
    # Fitting Random Forest Regression to the dataset 
    # import the regressor 
    from sklearn.ensemble import RandomForestRegressor 
    
    # create regressor object 
    model = RandomForestRegressor(n_estimators = 100, random_state = 0) 
    
    # fit the regressor with x and y data 
    model.fit(X_trainsc, y_train)
    
    y_pred = model.predict(X_testsc)
    
    
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - Random Forest')
    plt.legend()
    plt.show()
    
    
    from sklearn.metrics import r2_score
    y_pred_train = model.predict(X_trainsc)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
     
    print("\n--- \t{:0.3f} seconds --- Random Forest".format(time.time() - start_time_randForest))

def xgboostCalc():
    start_time_xgboost = time.time()
    
    global y_test, y_pred, y_train, X_test, X_testsc, X_train, X_trainsc
    
    # XGBoost
    import xgboost
    
#    for tuning parameters
#    parameters_for_testing = {
#        'colsample_bytree':[0.4,0.6,0.8],
#        'gamma':[0,0.03,0.1,0.3],
#        'min_child_weight':[1.5,6,10],
#        'learning_rate':[0.1,0.07],
#        'max_depth':[3,5],
#        'n_estimators':[1000],
#        'reg_alpha':[1e-5, 1e-2,  0.75],
#        'reg_lambda':[1e-5, 1e-2, 0.45],
#        'subsample':[0.6,0.95]  
#    }
#    
#                        
#    xgb_model = xgboost.XGBRegressor(learning_rate =0.1,
#                                     n_estimators=1000,
#                                     max_depth=5,
#                                     min_child_weight=1,
#                                     gamma=0,
#                                     subsample=0.8,
#                                     colsample_bytree=0.8,
#                                     nthread=6,
#                                     scale_pos_weight=1,
#                                     seed=42)
#    
#    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
#    gsearch1.fit(X_trainsc,y_train)
#    print (gsearch1.grid_scores_)
#    print('best params')
#    print (gsearch1.best_params_)
#    print('best score')
#    print (gsearch1.best_score_)
    
    
    best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                     gamma=0,                 
                     learning_rate=0.07,
                     max_depth=3,
                     min_child_weight=1.5,
                     n_estimators=1000,                                                                    
                     reg_alpha=0.75,
                     reg_lambda=0.45,
                     subsample=0.6,
                     seed=42)
    best_xgb_model.fit(X_trainsc,y_train)
    
    
    y_pred = best_xgb_model.predict(X_testsc)
    
    
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - XGBoost')
    plt.legend()
    plt.show()
    
    #from sklearn.metrics import r2_score
    y_pred_train = best_xgb_model.predict(X_trainsc)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %f" % (mae))
    
    mape = mean_absolute_percentage_error(y_test.reshape(y_test.shape[0]), y_pred.reshape(y_pred.shape[0]))
    print("MAPE: %.2f%%" % (mape))
    
    
    
    tscv = TimeSeriesSplit(n_splits=5)
#    for train_index, test_index in tscv.split(X):
#        print("train_index = " + str(max(train_index)))
#        print("test_index = " + str(max(test_index)))
#        print("---")
#        print("TRAIN:", train_index, "TEST:", test_index)
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
        
        
    scores = cross_val_score(best_xgb_model, X_trainsc, y_train, cv=tscv, scoring='r2')
    with np.printoptions(precision=3, suppress=True):
        print(scores)
    print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

    btscv = BlockingTimeSeriesSplit(n_splits=5)
    scores = cross_val_score(best_xgb_model, X_trainsc, y_train, cv=btscv, scoring='r2')    
    with np.printoptions(precision=3, suppress=True):
        print(scores)
    print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

    # Feature importance of XGBoost
    xgboost.plot_importance(best_xgb_model)
    #plt.rcParams['figure.figsize'] = [5, 5]
    # Calculate feature importances
    importances = best_xgb_model.feature_importances_
    # Sort feature importances in descending order
    #indices = np.argsort(importances)[::-1]
    indices = np.argsort(importances)[::]
    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]
    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.yticks(range(X.shape[1]), names, rotation=0)
    
    plt.show()
    
    
    print("\n--- \t{:0.3f} seconds --- XGBoost".format(time.time() - start_time_xgboost))

    start_time_xgboost2 = time.time()

    # Optimized structured data
    data_dmatrix = xgboost.DMatrix(data=X_trainsc,label=y_train)
    
    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                    'max_depth': 5, 'alpha': 10}
    
    cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=5,
                        num_boost_round=100,early_stopping_rounds=50,metrics="rmse", as_pandas=True, seed=42)
    
    print(cv_results.head())
    print((cv_results["test-rmse-mean"]).tail(1))
    
    print("\n--- \t{:0.3f} seconds --- XGBoost Cross-validation ".format(time.time() - start_time_xgboost2))

    
    aux_test = pd.DataFrame()    
    y_pred = np.float64(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    aux_test['error'] = y_test - y_pred
    aux_test['abs_error'] = aux_test['error'].apply(np.abs)
    aux_test['DEMAND'] = y_test
    aux_test['PRED'] = y_pred
    aux_test['Year'] = X_test['Year'].reset_index(drop=True)
    aux_test['Month'] = X_test['Month'].reset_index(drop=True)
    aux_test['Day'] = X_test['Day'].reset_index(drop=True)
    aux_test['Weekday'] = date.iloc[X_train.shape[0]:].dt.day_name().reset_index(drop=True)
    aux_test['Hour'] = X_test['Hour'].reset_index(drop=True)
    aux_test['Holiday'] = X_test['Holiday'].reset_index(drop=True)
    


    error_by_day = aux_test.groupby(['Year','Month','Day','Weekday', 'Holiday']) \
    .mean()[['DEMAND','PRED','error','abs_error']]

    print("\nOver forecasted days")
    print(error_by_day.sort_values(['error'], ascending=[False]).head(10))
    
    print("\nWorst absolute predicted days")
    print(error_by_day.sort_values('abs_error', ascending=False).head(10))
    
    print("\nBest predicted days")
    print(error_by_day.sort_values('abs_error', ascending=True).head(10))

    error_by_month = aux_test.groupby(['Year','Month']) \
    .mean()[['DEMAND','PRED','error','abs_error']]
    
    print("\nOver forecasted months")
    print(error_by_month.sort_values(['error'], ascending=[False]).head(10))
    
    print("\nWorst absolute predicted months")
    print(error_by_month.sort_values('abs_error', ascending=False).head(10))
    
    print("\nBest predicted months")
    print(error_by_month.sort_values('abs_error', ascending=True).head(10))
    

seasonDecomposeCalc()
featImportanceCalc()
#decisionTreeCalc()
#randForestCalc()
xgboostCalc()


import scipy.fftpack

# Number of samplepoints
N = y_test.shape[0]
# sample spacing
T = 1.0 / 24
x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf1 = scipy.fftpack.fft(y_test)
yf2 = scipy.fftpack.fft(y_pred)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf2[:N//2]), label = 'Predicted data')
plt.plot(xf, 2.0/N * np.abs(yf1[:N//2]), label = 'Real data')

plt.title('FFT')
plt.legend()
plt.show()


print("\n--- \t{:0.3f} seconds --- general processing".format(time.time() - start_time))



# the next command is the last line of my script
plt.ioff()
plt.show()