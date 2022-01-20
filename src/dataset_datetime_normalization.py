
import calendar
import datetime as dt
import seaborn as sns
import glob
import os
import pandas as pd
import numpy as np
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import time
start_time = time.time()

#from keras.layers import Dense, Activation
#from keras.models import Sequential


# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})

# os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask


# Importing the dataset
# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(
    f'\\{os.path.basename(__file__)}', '').replace('\\', '/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML', '')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src', '')

# Save all files in the folder
all_files = glob.glob(path + r'/datasets/ISONewEngland/csv-fixed/*.csv')

# Select datasets
# selectDatasets = ["2003","2004","2006","2007","2008","2009","2010","2011","2012","2013",
#              "2014","2015","2015","2016","2017","2018","2019"]
#selectDatasets = ["2009","2010","2011","2012","2013","2014","2015","2016"]
selectDatasets = ["2019"]

# Initialize dataset list
datasetList = []
holidayList = []


def normalizeDatetime():
    global dataset
    # Pre-processing input data
    # Verify zero values in dataset (X,y)
    print("Any null value in dataset?")
    print(dataset.isnull().any())
    print("How many are they?")
    print(dataset.isnull().sum())
    print("How many zero values?")
    print(dataset.eq(0).sum())
    print("How many zero values in y (DEMAND)?")
    print(dataset['DEMAND'].eq(0).sum())

    y = dataset.iloc[:, 3]

    # Taking care of missing data
    # if (dataset['DEMAND'].eq(0).sum() > 0):
    #     # Replace zero values by NaN
    #     dataset['DEMAND'].replace(0, np.nan, inplace= True)
    #     # Save y column (output)
    #     y = dataset.iloc[:, 3]
    #     # Replace NaN values by meaningful values
    #     from sklearn.preprocessing import Imputer
    #     y_matrix = y.as_matrix()
    #     y_matrix = y_matrix.reshape(y_matrix.shape[0],1)
    #     imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    #     imputer = imputer.fit(y_matrix)
    #     y =  imputer.transform(y_matrix)

    # Normalize hour
    dataset['HOUR'] = dataset['HOUR'] - 1

    # Decouple date and time from dataset
    # Then concat decoupled data
    date = pd.DataFrame()
    date = pd.to_datetime(dataset.DATE)
    date.dt.year.head()
    Year = pd.DataFrame({'Year': date.dt.year})
    Month = pd.DataFrame({'Month': date.dt.month})
    Day = pd.DataFrame({'Day': date.dt.day})
    Hour = pd.DataFrame({'HOUR': dataset.HOUR})

    #concatlist = [X,Year,Month,Day,Hour]
    #X = pd.concat(concatlist,axis=1)

    # DATASET CONVERTED to DATE + TIME
    test = pd.to_datetime(dataset.DATE)
    i = 0
    i2 = 0
    for row in test:
        test[i] = test[i] + pd.DateOffset(hours=0+i2)
        if (i2 == 23):
            i2 = 0
        else:
            i2 = i2 + 1
        i = i + 1
    print(test.head())
    df = pd.DataFrame(test)
    #concatlist = [X,df]
    #X = pd.concat(concatlist,axis=1)

    # Add weekday number to dataset
    dataset.drop(['DATE'], axis=1, inplace=True)
    concatlist = [df, dataset]
    dataset = pd.concat(concatlist, axis=1)

#    include = dataset['DATE'].loc['2009-01-01 00:00:00':'2009-12-31 23:00:00']
#    include = include.sort_values('DATE', ascending=True)


# Read all csv files and concat them
for filename in all_files:
    if (filename.find("hourly") != -1):
        for year in selectDatasets:
            if (filename.find(year) != -1):
                dataset = pd.read_csv(filename, index_col=None, header=0)
                normalizeDatetime()
                dataset.to_csv(filename.replace(
                    ".csv", "_fixed.csv"), index=None, header=True)
#                datasetList.append(df)

# Concat
#dataset = pd.concat(datasetList, axis=0, sort=False, ignore_index=True)
