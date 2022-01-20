# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:48:11 2020

@author: z003t8hn
"""
import os
#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
#import modin.pandas as pd
import pandas as pd

# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(
    f'\\{os.path.basename(__file__)}', '').replace('\\', '/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML', '')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src', '')
elif path.find('scripts') != -1:
    path = r'%s' % path.replace('/scripts', '')


outputPath=path + r'/datasets/ISONewEngland/not-fixed/'
outputPath = r'%s' % outputPath.replace('\\','/')

path = path + r'/xls/target/'

for root, dirs, files in os.walk(path):
    for filename in files:
        if ".csv" not in filename: 
            read_file = pd.ExcelFile(path + filename)
            sheetNames = read_file.sheet_names
            sheetNames.remove('Notes')
            
            df=read_file.parse(read_file.sheet_names)  # read a specific sheet to DataFrame
            
            for sheets in sheetNames:
                df[sheets].to_csv (outputPath + filename.replace(".xls","_"+sheets) + '.csv', index = None, header=True)
        
#        read_file = pd.read_csv(r'C:\Users\z003t8hn\code\ML\ML-Load-Forecasting\datasets\2016_smd_hourly.csv')