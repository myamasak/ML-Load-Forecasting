# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:48:11 2020

@author: z003t8hn
"""
import os
#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
#import modin.pandas as pd
import pandas as pd

dirPath='C:\\Users\\z003t8hn\\code\\ML\\ML-Load-Forecasting\\datasets\\'
dirPath = r'%s' % dirPath.replace('\\','/')
for root, dirs, files in os.walk(dirPath):
    for filename in files:
        if ".csv" not in filename: 
            read_file = pd.ExcelFile(dirPath + filename)
            sheetNames = read_file.sheet_names
            sheetNames.remove('Notes')
            
            df=read_file.parse(read_file.sheet_names)  # read a specific sheet to DataFrame
            
            for sheets in sheetNames:
                df[sheets].to_csv (dirPath + filename.replace(".xls","_"+sheets) + '.csv', index = None, header=True)
        
#        read_file = pd.read_csv(r'C:\Users\z003t8hn\code\ML\ML-Load-Forecasting\datasets\2016_smd_hourly.csv')