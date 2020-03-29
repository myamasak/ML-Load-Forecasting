# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import numpy as np
import pandas as pd

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
    

# Importing the dataset
path = r'%s' % os.getcwd().replace('\\','/')
#path = path + '/code/ML-Load-Forecasting'

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/*.csv')

# Select datasets 
#selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016"]
#selectDatasets = ["2015","2016","2017","2018","2019"]

# Initialize dataset list
datasetList = []


# Read all csv files and concat them
for filename in all_files:
    if (filename.find("ISONE") != -1):
        for data in selectDatasets:
            if (filename.find(data) != -1):
                df = pd.read_csv(filename,index_col=None, header=0)
                datasetList.append(df)

# Concat
dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)


## Pre-processing input data 
# Verify zero values in dataset (X,y)
print("Any null value in dataset?")
display(dataset.isnull().any())
print("How many are they?")
display(dataset.isnull().sum())
print("How many zero values?")
display(dataset.eq(0).sum())
print("How many zero values in y (DEMAND)?")
display(dataset['DEMAND'].eq(0).sum())


# Drop unnecessary columns in X dataframe (features)
X = dataset.iloc[:, :]
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP','DryBulb','DewPnt'], axis=1)
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegCP'], axis=1)
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','RT_LMP','RT_EC','RT_CC','RT_MLC'], axis=1)


y = dataset.iloc[:, 3]

# Taking care of missing data
if (dataset['DEMAND'].eq(0).sum() > 0):    
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
# Then concat decoupled data
date = pd.DataFrame() 
date = pd.to_datetime(dataset.Date)
date.dt.year.head() 
Year = pd.DataFrame({'Year':date.dt.year})
Month = pd.DataFrame({'Month':date.dt.month})
Day = pd.DataFrame({'Day':date.dt.day})
Hour = pd.DataFrame({'Hour':dataset.Hour})

concatlist = [X,Year,Month,Day,Hour]
X = pd.concat(concatlist,axis=1)

test = pd.to_datetime(dataset.Date)
i = 0
i2 = 0
for row in test:
    test[i] = test[i] + pd.DateOffset(hours=1+i2)  
#     print(test[i])
    if (i2 == 23):
         i2 = 0
    else:
        i2 = i2 + 1
    i = i + 1
print(test.head())
df = pd.DataFrame(test)
#concatlist = [X,df]
#X = pd.concat(concatlist,axis=1)



# Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(43)
from tensorflow import set_random_seed
set_random_seed(43)



# Splitting the dataset into the Training set and Test set
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
plt.show()


#from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
#import statsmodels.api as sm
data = pd.DataFrame(data=df)
concatlist = [data,pd.DataFrame(y)]
data = pd.concat(concatlist,axis=1)

data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data.drop(['index'], axis=1)
result = seasonal_decompose(data, model='multiplicative')
#result = sm.tsa.seasonal_decompose(data)
result.plot()
plt.show



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
plt.title('Prediction')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
y_pred_train = model.predict(X_trainsc)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))


# import export_graphviz 
#from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
#export_graphviz(model, out_file ='tree.dot', 
#               feature_names = X.columns.values.tolist()) 
#


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
plt.title('Prediction')
plt.legend()
plt.show()


#from sklearn.metrics import r2_score
y_pred_train = model.predict(X_trainsc)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))



