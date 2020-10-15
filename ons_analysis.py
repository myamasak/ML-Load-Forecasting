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
import BlockingTimeSeriesSplit as btss
import matplotlib.pyplot as plt


# Print configs
pd.options.display.max_columns = None
pd.options.display.width=1000

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')

# Save all files in the folder
filename = glob.glob(path + r'/datasets/ONS/*.csv')
filename = filename[0].replace('\\','/')
dataset = pd.read_csv(filename,index_col=None, header=0, delimiter=";")

# Selection of year
selectDatasets = ["2011","2012","2013","2014","2015","2016","2017","2018"]

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

# Taking care of missing data
if (dataset['Demanda'].eq(0).sum() > 0
    or dataset['Demanda'].isnull().any()):    
    print(dataset[dataset['Demanda'].isnull()])
    # Save the NaN indexes
    nanIndex = dataset[dataset['Demanda'].isnull()].index.values
    # Replace zero values by NaN
    dataset['Demanda'].replace(0, np.nan, inplace=True)
    # Save y column (output)
#    col = dataset.columns.get_loc('Demanda')
#    y = dataset.iloc[:, col]
    # Replace NaN values by meaningful values
#    from sklearn.impute import SimpleImputer
#    y_matrix = y.to_numpy()
#    y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
#    # imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#    imputer = imputer.fit(y_matrix)
#    y = imputer.transform(y_matrix)
#    y = y.reshape(y.shape[0])
    
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

# Dummy Enconding
# X = pd.get_dummies(X, columns=['Year'], drop_first=False, prefix='Year')
# X = pd.get_dummies(X, columns=['Month'], drop_first=False, prefix='Month')
# X = pd.get_dummies(X, columns=['Weekday'], drop_first=False, prefix='Wday')



# Split X data to different subsystems/regions
Xs = X[X['Subsistema'].str.find("Sul") != -1].reset_index(drop=True)
Xs = Xs.drop(['Subsistema','Data'],axis=1)
#Xn = X[X['Subsistema'].str.find("Norte") != -1].reset_index(drop=True)
#Xn = Xn.drop(['Subsistema','Data'],axis=1)
#Xne = X[X['Subsistema'].str.find("Nordeste") != -1].reset_index(drop=True)
#Xne = Xne.drop(['Subsistema','Data'],axis=1)
#Xse = X[X['Subsistema'].str.find("Sudeste/Centro-Oeste") != -1].reset_index(drop=True)
#Xse = Xse.drop(['Subsistema','Data'],axis=1)
#Xall = X[X['Subsistema'].str.find("Todos") != -1].reset_index(drop=True)
#Xall = Xall.drop(['Subsistema','Data'],axis=1)

# Split y data to different subsystems/regions
ys = y[y['Subsistema'].str.find("Sul") != -1]['Demanda'].reset_index(drop=True)
yn = y[y['Subsistema'].str.find("Norte") != -1]['Demanda'].reset_index(drop=True)
yne = y[y['Subsistema'].str.find("Nordeste") != -1]['Demanda'].reset_index(drop=True)
yse = y[y['Subsistema'].str.find("Sudeste/Centro-Oeste") != -1]['Demanda'].reset_index(drop=True)
yall = y[y['Subsistema'].str.find("Todos") != -1]['Demanda'].reset_index(drop=True)

y = pd.concat([ys, yn, yne, yse, yall], axis=1)
y.columns = ['South','North','NorthEast','SouthEast','All Regions']

# Save in Date format
df = X[X['Subsistema'].str.find("Sul") != -1]['Data'].reset_index(drop=True)

# Plot all data to general analysis
plt.figure()
plt.plot(df,ys,df,yn,df,yne,df,yse,df,yall)
#plt.scatter(Xs,ys)
#plt.scatter(Xn,yn)
#plt.scatter(Xne,yne)
#plt.scatter(Xse,yse)
#plt.scatter(Xall,yall)
plt.title('Demand')
plt.legend(['South','North','NorthEast','SouthEast','All Regions'])
plt.tight_layout()
plt.show()
plt.savefig('ONS_all_Regions_Demand_plot')

# Plot south data only
plt.figure()
plt.plot(df,yall)
plt.title('Demand of all regions')
plt.tight_layout()
plt.show()
plt.savefig('ONS_All_Demand_plot')

# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(42)

from tensorflow import set_random_seed
set_random_seed(42)

Xs = pd.DataFrame(Xs)
#ys = pd.DataFrame(ys)

# Splitting the dataset into the Training set and Test set
# Forecast Ndays - Calculate testSize in percentage
Ndays = 120
testSize = (Ndays*24)/(ys.shape[0])
#testSize = 0.1
X_train, X_test, y_train, y_test = train_test_split(Xs, yall, test_size = testSize, random_state = 0, shuffle = False)

y_ = pd.concat([y_train, y_test])
X_ = pd.concat([X_train, X_test])

def outlierDetection(y_, columnName):
    # global X_train, X_test, y_train, y_test, X_
    
    import plotly.io as pio
    import plotly.graph_objects as go
    # import plotly
    pio.renderers.default = 'browser'
    pio.kaleido.scope.default_width = 1200
    pio.kaleido.scope.default_height = 750
    
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=20)

    y_pred = clf.fit_predict(pd.DataFrame(y_))
#    outliers_train = y_train.loc[y_pred_train == -1]
    
    negativeOutlierFactor = clf.negative_outlier_factor_
    outliers = y_.loc[negativeOutlierFactor < (negativeOutlierFactor.mean() - negativeOutlierFactor.std()-1)]
    
#    outliers.reindex(list(range(outliers.index.min(),outliers.index.max()+1)),fill_value=0)
    

    outliers_reindex = outliers.reindex(list(range(df.index.min(),df.index.max()+1)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df,
                            y=y_,
                            name=columnName,
                            showlegend=False,
                            mode='lines'))                         
    fig.add_trace(go.Scatter(x=df,
                            y=outliers_reindex,
                            name='Outliers',
                            mode='markers',
                            marker_size=10))
    # Edit the layout
    fig.update_layout(title=columnName+' Demand outliers',
                    xaxis_title='Date',
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
    fig.write_image("ONS_outliers_"+columnName+".svg")
    
    # Fix outliers by removing and replacing with interpolation
    y_ = pd.DataFrame(y_).replace([outliers],np.nan)    
    y_ = y_.interpolate(method='linear', axis=0).ffill().bfill()
    
    print('Outliers fixed: ', end='\n')
    print(y_.loc[outliers.index.values], end='\n')
    
    # Transform to numpy arrays    
    y_ = np.array(y_)
    y_ = y_.reshape(y_.shape[0])
    
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
                    xaxis_title='Date',
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
    fig.write_image("ONS_outliers_fixed_"+columnName+".svg")
    
    return y_

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def seasonDecomposeCalc(autoCorrelation):
    print("Running Seasonal Decompose calculation...")
    start_time_seasonDecompose = time.time()
    
    #from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
    #import statsmodels.api as sm
    data = pd.DataFrame(data=df)
    concatlist = [data,pd.DataFrame(y_)]
    data = pd.concat(concatlist,axis=1)
    
    data.reset_index(drop=True,inplace=True)
    data['Data'] = pd.to_datetime(data['Data'])
    data = data.set_index('Data')
    
    # data.sort_values('DEMAND', ascending=False).head(10)
    
    data[data.isnull().any(axis=1)]
    
    
    # result = seasonal_decompose(data.round(0), model='multiplicative', freq=24, extrapolate_trend='freq')
    result = seasonal_decompose(data, model='multiplicative', freq=24)
    result.plot()
    plt.tight_layout()
    plt.show()
    plt.savefig('ONS_seasonal_decompose.png')
    
#    plt.figure()
#    plt.plot(result.trend)
#    plt.title('Trend')
#    plt.show()
    
    detrended = y_ - result.trend['Demanda'].reset_index(drop=True)
    plt.figure()
    plt.plot(detrended)
    plt.title('Detrended')
    plt.show()
    
#    plt.figure()
#    plt.plot(result.seasonal)
#    plt.title('Seasonal')
#    plt.show()
    
    deseasonalized = y_ - result.seasonal['Demanda'].reset_index(drop=True)
    plt.figure()
    plt.plot(deseasonalized)
    plt.title('Deseasonalized')
    plt.show()
    
    if autoCorrelation:
        from pandas.plotting import autocorrelation_plot
        
        # Draw Plot
        plt.figure()
        autocorrelation_plot(y_)
        plt.show()
        
    #    from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
    #    plot_acf(y_)
    #    plot_pacf(y_, lags=50)
        
        # PACF plot of 1st differenced series
    #    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(pd.DataFrame(y_).diff()); axes[0].set_title('1st Differencing')
    #    axes[1].set(ylim=(0,5))
        plot_acf(pd.DataFrame(y_).diff().dropna(), ax=axes[1])
        plt.show()
    

    
    print("\n--- \t{:0.3f} seconds --- Seasonal Decompose".format(time.time() - start_time_seasonDecompose))

def featImportanceCalc():
    print("Running Feature Importance (RF) calculation...")
    
    start_time_featImportance = time.time()
    
    ## Feature importance
    # Import random forest
    #from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor  
    
    # Create decision tree classifer object
    #clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf = RandomForestRegressor(random_state=0, n_jobs=-1)
    
    # Train model
    model = clf.fit(Xs, y_)
    
    # Calculate feature importances
    importances = model.feature_importances_    
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::]    
    # Rearrange feature names so they match the sorted feature importances
    names = [Xs.columns[i] for i in indices]
    
    # make a plot with the feature importance
    # plt.figure(figsize=(12,14), dpi= 80, facecolor='w', edgecolor='k')
    plt.figure()
    # plt.grid()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], height=0.2, align='center')
#    plt.axvline(x=0.25)
    plt.yticks(range(len(indices)), list(names))
    plt.xlabel('Relative Importance')   
    
    # Show plot
    plt.show()
    plt.savefig('ONS_Feature_Importance_RF.png')
    
    featImportance = pd.concat([pd.DataFrame({'Features':names}),
                  pd.DataFrame({'Relative_Importance':importances[indices]})],
                    axis=1, sort=False)
    
    print(featImportance)
    
    print("\n--- \t{:0.3f} seconds --- Feature Importance".format(time.time() - start_time_featImportance))




def decisionTreeCalc():
    print("Running Decision Tree calculation...")
    start_time_decisionTree = time.time()

    # import the regressor 
    from sklearn.tree import DecisionTreeRegressor 
    
    # create a regressor object 
    model = DecisionTreeRegressor(random_state = 42) 
    
    # fit the regressor with X and Y data 
    #model.fit(X, y) 
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)    
    
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y_, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - Decision Tree')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('ONS_DecisionTree_pred.png')
    
    
    y_pred_train = model.predict(X_train)
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

def randForestCalc(enableLearningCurve):
    print("Running Random Forest calculation...")
    start_time_randForest = time.time()
    # Fitting Random Forest Regression to the dataset 
    # import the regressor 
    from sklearn.ensemble import RandomForestRegressor 
    
    # create regressor object 
    model = RandomForestRegressor(n_estimators = 100, random_state = 0) 
    
    # fit the regressor with x and y data 
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    
    rows = X_test.index
    
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y_, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - Random Forest')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('ONS_RandomForest_pred.png')
    
    
    from sklearn.metrics import r2_score
    y_pred_train = model.predict(X_train)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    
    if enableLearningCurve:
        # Create CV training and test scores for various training set sizes
        train_sizes, train_scores, test_scores = learning_curve(model,
                                                   X_train, y_train, cv=5, scoring='r2', n_jobs=-1,
                                                   shuffle=False, random_state=42,
                                                   # 50 different sizes of the training set
                                                   train_sizes=np.linspace(0.001, 1.0, 50))
    
        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
    
        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
    
        # Draw lines
        plt.subplots(1, figsize=(7,7))
        plt.plot(train_sizes, train_mean, '--',  label="Training score")
        plt.plot(train_sizes, test_mean, label="Cross-validation score")
    
        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
        # Create plot
        plt.title("Random Forest - Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("RMSE Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.ylim([0,1])
        plt.show()
        plt.savefig('ONS_RandomForest_learningcurve.png')
     
    print("\n--- \t{:0.3f} seconds --- Random Forest".format(time.time() - start_time_randForest))

def xgboostCalc(enableCV, enableLearningCurve):
    print("Running XGBoost calculation...")
    start_time_xgboost = time.time()
    
    global y_test, y_pred, y_train, X_test, X_testsc, X_train, X_trainsc
    
    # XGBoost
    import xgboost

    X_trainsc = X_train
    X_testsc = X_test
    
    
    eval_set = [(X_trainsc, y_train), (X_testsc, y_test)]

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
    model.fit(X_trainsc,y_train,eval_metric=["rmse", "mae"],eval_set=eval_set, verbose=False)
    
    
    y_pred = model.predict(X_testsc)
    
    
    rows = X_test.index
    df2 = df.iloc[rows[0]:]
    
    plt.figure()
    #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
    plt.plot(df,y_, label = 'Real data')
    plt.plot(df2,y_pred, label = 'Predicted data')
    plt.title('Prediction - XGBoost')
    plt.legend()
    plt.show()
    plt.savefig('XGBoost_pred.png')
    
    #from sklearn.metrics import r2_score
    y_pred_train = model.predict(X_trainsc)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %f" % (mae))
    
    mape = mean_absolute_percentage_error(y_test.to_numpy(), y_pred)
    print("MAPE: %.2f%%" % (mape))
    
    
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('epochs')
    plt.title('XGBoost RMSE')
    plt.show()
    plt.savefig('XGBoost_RMSE.png')

    # plot classification error
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Test')
    ax.legend()
    plt.ylabel('MAE')
    plt.xlabel('epochs')
    plt.title('XGBoost MAE')
    plt.show()
    plt.savefig('XGBoost_MAE.png')
    
    print("\n--- \t{:0.3f} seconds --- XGBoost".format(time.time() - start_time_xgboost))

    if enableCV:
        start_time_xgboost2 = time.time()
        
        tscv = TimeSeriesSplit(n_splits=5)
#        for train_index, test_index in tscv.split(X):
#            print("train_index = " + str(max(train_index)))
#            print("test_index = " + str(max(test_index)))
#            print("---")
#            print("TRAIN:", train_index, "TEST:", test_index)
#            X_train, X_test = X[train_index], X[test_index]
#            y_train, y_test = y[train_index], y[test_index]
            
        print("Running XGBoost CrossValidation Time Series Split...")
        scores = cross_val_score(model, X_trainsc, y_train, cv=tscv, scoring='r2')
        with np.printoptions(precision=4, suppress=True):
            print(scores)
        print("Loss: {0:.6f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

#        print("Running XGBoost CrossValidation Blocking Time Series Split...")
#        btscv = btss.BlockingTimeSeriesSplit(n_splits=5)
#        scores = cross_val_score(model, X_trainsc, y_train, cv=btscv, scoring='r2')    
#        with np.printoptions(precision=4, suppress=True):
#            print(scores)
#        print("Loss: {0:.6f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
#
        print("\n--- \t{:0.3f} seconds --- XGBoost Cross-validation ".format(time.time() - start_time_xgboost2))

        # Feature importance of XGBoost
        xgboost.plot_importance(model)
        #plt.rcParams['figure.figsize'] = [5, 5]
        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        #indices = np.argsort(importances)[::-1]
        indices = np.argsort(importances)[::]
        # Rearrange feature names so they match the sorted feature importances
        names = [Xs.columns[i] for i in indices]
        # Add bars
        plt.bar(range(Xs.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.yticks(range(Xs.shape[1]), names, rotation=0)
        
        plt.show()
        plt.savefig('feature_importance_xgboost.png')
        
        
        # print("\n--- \t{:0.3f} seconds --- XGBoost".format(time.time() - start_time_xgboost))

        # start_time_xgboost2 = time.time()

        # # Optimized structured data
        # data_dmatrix = xgboost.DMatrix(data=X_trainsc,label=y_train)
        
        # params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
        #                 'max_depth': 5, 'alpha': 10}
        
        # cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=5,
        #                     num_boost_round=100,early_stopping_rounds=50,metrics="rmse", as_pandas=True, seed=42)
        
        # print(cv_results.head())
        # print((cv_results["test-rmse-mean"]).tail(1))
        
#        print("\n--- \t{:0.3f} seconds --- XGBoost Cross-validation ".format(time.time() - start_time_xgboost2))

        
        aux_test = pd.DataFrame()    
        y_pred = np.float64(y_pred)
        y_pred = y_pred.reshape(y_pred.shape[0])
        try:
            y_test = y_test.reshape(y_test.shape[0])
        except Exception:
            print(Exception)
        aux_test['error'] = y_test.reset_index(drop=True) - y_pred
        aux_test['abs_error'] = aux_test['error'].apply(np.abs)
        aux_test['DEMAND'] = y_test.reset_index(drop=True)
        aux_test['PRED'] = y_pred
        aux_test['Year'] = X_test['Year'].reset_index(drop=True)
        aux_test['Month'] = X_test['Month'].reset_index(drop=True)
        aux_test['Day'] = X_test['Day'].reset_index(drop=True)
        aux_test['Weekday'] = df.iloc[X_test.index.values].dt.day_name().reset_index(drop=True)
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
        
        
    if enableLearningCurve:
        print("Running XGBoost Learning Curve...")
        start_time_xgboost3 = time.time()
        
        # Create CV training and test scores for various training set sizes
        train_sizes, train_scores, test_scores = learning_curve(model,
                                                X_trainsc, y_train, cv=5, scoring='r2', n_jobs=-1,
                                                # 50 different sizes of the training set
                                                train_sizes=np.linspace(0.01, 1.0, 50))

        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.subplots(1, figsize=(7,7))
        plt.plot(train_sizes, train_mean, '--',  label="Training score")
        plt.plot(train_sizes, test_mean, label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        plt.title("XGBoost - Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("RMSE Score"), plt.legend(loc="best")
        plt.tight_layout(); plt.show()
        plt.savefig('XGBoost_learningcurve.png')


        
    print("\n--- \t{:0.3f} seconds --- XGBoost Learning curve".format(time.time() - start_time_xgboost3))

################
# MAIN PROGRAM
################
    
# loop over all demand regions / y[columns]
for column in y:
    y[column] = outlierDetection(y_ = y[column], columnName = column)
    # X_train, X_test, y_train, y_test = train_test_split(Xs, yall, test_size = testSize, random_state = 0, shuffle = False)
    y_train, y_test = train_test_split(y[column], test_size = testSize, random_state = 0, shuffle = False)

    seasonDecomposeCalc()
    decisionTreeCalc()
    randForestCalc(enableLearningCurve=True)
    xgboostCalc(enableCV=True, enableLearningCurve=True)
