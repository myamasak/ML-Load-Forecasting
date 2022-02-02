"""
Diebold-Mariano statistical test
Author: Marcos Yamasaki
14/01/2022
"""

from dm_test.dm_test import dm_test
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
import time
import seaborn as sns

start_time = time.time()
plt.close("all")
register_matplotlib_converters()
sys.path.append('../')

# CONSTANTS
DATASET_NAME = 'ons'
ALGORITHM = 'xgboost'
FORECASTDAYS = 7
STEPS_AHEAD = 24*1
NMODES = 1
MODE = 'ceemdan'
selectDatasets = ["2015", "2016", "2017", "2018"]

# seaborn configuration
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

if False:
    y_test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    np.random.seed(15)
    y_pred1 = np.random.randint(1,17,len(y_test))
    y_pred2 = y_pred1-2
    #y_pred2 = np.random.randint(1,17,len(y_test))

    r = dm_test(y_test, y_pred1, y_pred2, h=1, crit="MSE")
    print("MSE")
    print(r)
    r = dm_test(y_test, y_pred1, y_pred2, h=1, crit="MAD")
    print("MAD")
    print(r)
    r = dm_test(y_test, y_pred1, y_pred2, h=1, crit="MAPE")
    print("MAPE")
    print(r)
    r = dm_test(y_test, y_pred1, y_pred2, h=1, crit="poly")
    print("poly")
    print(r)

    plt.plot(y_test, label='y_test')
    plt.plot(y_pred1, label='y_pred1')
    plt.plot(y_pred2, label='y_pred2')
    plt.legend()


try:  
    all_files = glob.glob(
                    path + f"/datasets/{DATASET_NAME}/y_pred/*.csv")
    # path+f'/datasets/{DATASET_NAME}/y_pred/y_pred_{ALGORITHM}_{MODE}-{NMODES}_forecast{STEPS_AHEAD}_{selectDatasets[0]}-{selectDatasets[-1]}.csv', index=None, header=['y_test','y_pred'])
    # Initialize dataset list
    concat = []
    # Read all csv files and concat them
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        concat.append(df)

except (AttributeError, ValueError, KeyError, UnboundLocalError, FileNotFoundError, ValueError, OSError, IOError) as e:
    print(e)
    raise


  