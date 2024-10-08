"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import numpy as np
import pandas as pd
import glob

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})


# Set path to import dataset and export figures
try:
    path = os.path.realpath(__file__)
    path = r'%s' % path.replace(f'\\{os.path.basename(__file__)}','').replace('\\','/')
    if path.find('autoML') != -1:
        path = r'%s' % path.replace('/autoML','')
    elif path.find('src') != -1:
        path = r'%s' % path.replace('/src','')
except NameError:
    path = os.getcwd()
    path = path.replace('\\','/').replace('src','')
# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONE/csv-fixed/*.csv') + \
            glob.glob(path + r'/datasets/ISONE/holidays/*.csv')

# Select datasets 
#selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
#selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016","2017"]
selectDatasets = ["2015","2016","2017","2018"]

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
#holidays.drop_duplicates(subset=['DATE'], keep=False, inplace=True)
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
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP','DRYBULB','DEWPNT'], axis=1)
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','DATE','HOUR','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','RT_LMP','RT_EC','RT_CC','RT_MLC'], axis=1)


y = dataset.iloc[:, 3]

# Taking care of missing data
if (dataset['DEMAND'].eq(0).sum() > 0
    or dataset['DEMAND'].isnull().any()):    
    print(dataset['DEMAND'][dataset['DEMAND'].isnull()])
    # Save the NaN indexes
    nanIndex = dataset[dataset['DEMAND'].isnull()].index.values
    # Replace zero values by NaN
    dataset['DEMAND'].replace(0, np.nan, inplace=True)
    #convert to float
    y = dataset['DEMAND'].astype(float)
    y = y.interpolate(method='linear', axis=0).ffill().bfill()
    print(y.iloc[nanIndex])



date = pd.DataFrame() 
date = pd.to_datetime(dataset.DATE)
date.dt.year.head() 
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'HOUR':dataset.HOUR})

concatlist = [X,Year,Month,Day,Hour]
X = pd.concat(concatlist,axis=1)

df = date

# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)



plt.figure()
plt.plot(df,y, color = 'gray', label = 'Real data')
plt.legend()
plt.show()


#from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
#import statsmodels.api as sm
data = pd.DataFrame(data=df)
concatlist = [data,pd.DataFrame(y)]
data = pd.concat(concatlist,axis=1)

data.reset_index(inplace=True)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')
data = data.drop(['index'], axis=1)
#data.columns = ['DATE','DEMAND']
data.columns = ['DEMAND']
result = seasonal_decompose(data, period=24, model='additive', extrapolate_trend='freq')
result2 = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
#result = sm.tsa.seasonal_decompose(data)
result.plot()
plt.show


#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(3)
rolling_mean = data.rolling(window = 12).mean()
rolling_std = data.rolling(window = 12).std()
plt.plot(data, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()


result2 = adfuller(data.iloc[:,0].values, regression='ct', maxlag=168)
print('ADF Statistic: {}'.format(result2[0]))
print('p-value: {}'.format(result2[1]))
pvalue = result2[1]
print('Critical Values:')
for key, value in result2[4].items():
    print('\t{}: {}'.format(key, value))
print(f'Used lags: {result2[2]}')
print(f'Number of observations: {result2[3]}')

if pvalue < 0.05:
    print(f'p-value < 0.05, so the series is stationary.')
else:
    print(f'p-value > 0.05, so the series is non-stationary.')
    
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# from scipy import stats
# seasonal_test = stats.kruskal(data.iloc[:,0].values)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


print("ACF plot of demand series")
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(pd.DataFrame(y)); axes[0].set_title('Demand')
axes[1].set(ylim=(-0.5,1.2))
#axes[1].set(xlim=(-10,100))
plot_acf(pd.DataFrame(y).dropna(), ax=axes[1], lags=24*360)
plt.show()

sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')


import scipy
m, lower, upper = mean_confidence_interval()

from matplotlib.collections import PolyCollection, LineCollection

fig, ax = plt.subplots()
plot_pacf(y, lags=24*7, ax=ax, method="ywm", use_vlines=True, marker='', vlines_kwargs={'color': 'black', 'linewidth': 0.5})
for item in ax.collections:
    #change the color of the CI 
    if type(item)==PolyCollection:
        item.set_facecolor('red')
        item.set_alpha(1)
plt.show()

# ACF plot of 1st differenced series
print("ACF plot of 1st differenced series")
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(pd.DataFrame(y).diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(-0.5,1.2))
#axes[1].set(xlim=(-10,100))
plot_acf(pd.DataFrame(y).diff().dropna(), ax=axes[1], lags=np.arange(len(pd.DataFrame(y).diff().dropna())))
plt.show()



print("ACF plot of 2nd differenced series")
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(pd.DataFrame(y).diff().diff()); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(-0.5,1.2))
axes[1].set(xlim=(-10,100))
plot_acf(pd.DataFrame(y).diff().diff().dropna(), ax=axes[1], lags=100)

plt.show()


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


#from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
#freq : str {'B','D','W','M','A', 'Q'}
#    'B' - business day, ie., Mon. - Fri.
#    'D' - daily
#    'W' - weekly
#    'M' - monthly
#    'A' - annual
#    'Q' - quarterly
model = ARIMA(data, order=(2,1,1))
model_fit = model.fit()
print(model_fit.summary())


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
#model_fit.plot_predict(dynamic=False)
#plt.show()


from statsmodels.tsa.stattools import acf



# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


data_diff = data.diff().dropna()

# Create Training and Test
# Splitting the dataset into the Training set and Test set
forecastDays = 7
testSize = forecastDays*24/data.shape[0]
train, test = train_test_split(data_diff, test_size=testSize, random_state=42, shuffle = False)


# Build Model
# model = ARIMA(train, order=(3,2,1))  
model = ARIMA(train, order=(27, 0, 11))  
model_fit = model.fit()
print(model_fit.summary())

# Forecast
forecast = model_fit.predict(start=test.index[0], end=test.index[-1])

#forecast_inv = np.concatenate([data.iloc[0].values, forecast]).cumsum()


#nIndex = test.index.size
#fc, se, conf = fitted.forecast(nIndex, alpha=0.05)  # 95% conf

# Make as pandas series
#fc_series = pd.Series(fc, index=test.index[:nIndex])
#lower_series = pd.Series(conf[:, 0], index=test.index[:nIndex])
#upper_series = pd.Series(conf[:, 1], index=test.index[:nIndex])

r2test = r2_score(test.values.ravel(), forecast.values.ravel())
print(f"r2test: {r2test:0.5f}")
n = len(test)
p = test.shape[1]
adjr2_score= 1-((1-r2test)*(n-1)/(n-p-1))
print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adjr2_score))

rmse = np.sqrt(mean_squared_error(test.values.ravel(), forecast.values.ravel()))
print("RMSE: %f" % (rmse))

mae = mean_absolute_error(test.values.ravel(), forecast.values.ravel())
print("MAE: %f" % (mae))

try:
    y_test = y_test.values.reshape(test.shape[0])
    mape = mean_absolute_percentage_error(test.values.ravel(), forecast.values.ravel())
    smape = symmetric_mape(test.values.ravel(), forecast.values.ravel())
except (AttributeError,ValueError) as e:
    mape = mean_absolute_percentage_error(test.values.ravel(), forecast.values.ravel())
    smape = symmetric_mape(test.values.ravel(), forecast.values.ravel())
print("MAPE: %.2f%%" % (mape))
print("sMAPE: %.2f%%" % (smape))

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(forecast, label='forecast')
#plt.fill_between(lower_series.index, lower_series, upper_series, 
#                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()




data.hist()
plt.show()
data.plot(kind='kde')
plt.show()




plt.figure()
plt.plot(data,linewidth=1)
#plt.plot(data,marker='.', alpha=0.5, linestyle='None')

#data.reset_index(drop=True,inplace=True)
Month.reset_index(inplace=True)
#data.reset_index(inplace=True)
Month['DATE'] = pd.to_datetime(data.index)
Month = Month.set_index('DATE')
Month = Month.drop(['index'], axis=1)
concatlist = [data,Month]
newData = pd.concat(concatlist,axis=1,sort=False)


plt.figure()
sns.boxplot(data=newData, x='Month', y='DEMAND')
plt.ylabel('DEMAND')

    

#import pmdarima as pm

#
#model = pm.auto_arima(train, start_p=1, start_q=1,
#                      test='adf',       # use adftest to find optimal 'd'
#                      max_p=6, max_q=6, # maximum p and q
#                      m=24,              # frequency of series
#                      d=None,           # let model determine 'd'
#                      seasonal=True,   # Yes Seasonality
#                      start_P=0, 
#                      D=0, 
#                      trace=True,
#                      error_action='ignore',  
#                      suppress_warnings=True, 
#                      stepwise=True)
#
#print(model.summary())
#
from statsmodels.tsa.arima.model import SARIMAX
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(train,order=(6,1,6),seasonal_order=(1,0,2,24),
                                  enforce_stationarity=False, enforce_invertibility=False)

#fitted = model.fit(disp=-1)
fitted = model.fit(train)
#print(fitted.summary())

# Forecast
nIndex = test.index.size
#fc, se, conf = fitted.forecast(nIndex, alpha=0.05)  # 95% conf
#y_pred = fitted.predict(test.index[0],test.index[-1])
y_pred = fitted.predict(test.shape[0])

# Make as pandas series
fc_series = pd.Series(y_pred, index=test.index[:nIndex])
lower_series = pd.Series(conf[:, 0], index=test.index[:nIndex])
upper_series = pd.Series(conf[:, 1], index=test.index[:nIndex])
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

model.plot_diagnostics()


## Feature importance
# Import random forest
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor  


# Replace NaN values by meaningful values
#from sklearn.preprocessing import Imputer
#y_matrix = Xdata['RegSP'].as_matrix()
#y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
#imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer = imputer.fit(y_matrix)
#Xdata['RegSP'] =  imputer.transform(y_matrix)
# Taking care of missing data
print('Taking care of missing data')
if (dataset['DEMAND'].eq(0).sum() > 0
    or dataset['DEMAND'].isnull().any()):    
    print(dataset[dataset['DEMAND'].isnull()])
    # Save the NaN indexes
    nanIndex = dataset[dataset['DEMAND'].isnull()].index.values
    # Replace zero values by NaN
    dataset['DEMAND'].replace(0, np.nan, inplace=True)
    #convert to float
    y = dataset['DEMAND'].astype(float)
    y = y.interpolate(method='linear', axis=0).ffill().bfill()
    print(y.iloc[nanIndex])


# Create decision tree classifer object
#clf = RandomForestClassifier(random_state=0, n_jobs=-1)
clf = RandomForestRegressor(random_state=0, n_jobs=-1)

Xdata = dataset.iloc[:, :]
Xdata = Xdata.drop(['DATE','HOUR','DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','SYSLoad'], axis=1)

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

#def plot_feature_importances(importances, features):
#    # get the importance rating of each feature and sort it
#    indices = np.argsort(importances)
#    
#    # make a plot with the feature importance
#    plt.figure(figsize=(12,14), dpi= 80, facecolor='w', edgecolor='k')
#    plt.grid()
#    plt.title('Feature Importances')
#    plt.barh(range(len(indices)), importances[indices], height=0.8, color='mediumvioletred', align='center')
#    plt.axvline(x=0.03)
#    plt.yticks(range(len(indices)), list(features[indices]))
#    plt.xlabel('Relative Importance')
#    plt.show()










