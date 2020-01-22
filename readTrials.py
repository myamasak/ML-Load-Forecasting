# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:29:23 2020

@author: z003t8hn
"""
import os


class Trials:
    def __init__(self, paramid, jobid, trialtype, sequence, value):
        self.paramid = paramid
        self.jobid = jobid
        self.trialtype = trialtype
        self.sequence = sequence
        self.value = value
        
        
# Basically, it reads all subdirectories of dirPath, find for metrics file
# and bring the contents to python.
# Some string handling is done in the way, in order to remove unnecessary strings.
# The final result is a object list (Trials list), where it's possible to do some
# statistics and evaluations with the dataset

def importMetricsData():    
    dirPath="C:\\Users\\z003t8hn\\nni\\experiments\\LiNYuulo"
    filename = "metrics"    
    for root, dirs, files in os.walk(dirPath):
         for file in files:
             if (file==filename):
                 with open(os.path.join(root, file), "r") as auto:            
    #                print(auto.readline())
                    line = auto.readline()
                    if line:
                        data = line.split("\"")
                        for index in data:
                            if index == ": ":
                                data.remove(": ")
                            if index == ":":
                                data.remove(":")
                            if index == ", ":
                                data.remove(", ")
                            if index == ",":
                                data.remove(",")
                            if index.find("{") != -1:
                                data.pop(0)                    
                             
                        for idx,index in enumerate(data):
                            if ((index.find(":") != -1) or
                                (index.find(",") != -1) or
                                (index.find("}") != -1)):
                                indexStr = data[idx].strip(" :,}\n")
                                data.insert(idx,indexStr)
                                data.pop(idx+1)
                        
                        print(data)
                        trialObj = Trials(data[1],data[3],data[5],data[7],data[9])
                        trialList.append(trialObj)
                    

# main()
trialList = []
importMetricsData()

import pandas as pd 
import numpy as np

dataset = pd.DataFrame(columns=['paramid','jobid','trialtype','sequence','value'])

for i1,trials in enumerate(trialList):
    # Pass the row elements as key value pairs to append() function 
    dataset = dataset.append({'paramid': trials.paramid,
                                'jobid': trials.jobid,
                                'trialtype': trials.trialtype,
                                'sequence': trials.sequence,
                                'value': trials.value                                  
                                } , ignore_index=True)


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)


#x = np.random.normal(size=100)
datanp=dataset.value.to_numpy().astype(float)
plt.figure()
sns.distplot(datanp)
plt.show()
plt.figure()
sns.distplot(datanp, kde=False, rug=True);
plt.show()
plt.figure()
sns.distplot(datanp, bins=20, kde=False, rug=True);
plt.show()

plt.figure()
sns.distplot(datanp, hist=False, rug=True);
plt.show()  

plt.figure()
sns.kdeplot(datanp, shade=True);
plt.show()  

plt.figure()
sns.kdeplot(datanp)
sns.kdeplot(datanp, bw=.01, label="bw: 0.01")
sns.kdeplot(datanp, bw=1, label="bw: 1")
plt.legend();
plt.show()  

# plt.figure(1)
# #plt.plot(df2,y_tested, color = 'red', label = 'Real data')
# plt.plot(df,y, color = 'gray', label = 'Real data')
# plt.plot(df2,y_pred, color = 'blue', label = 'Predicted data')
# plt.title('Prediction')
# plt.legend()
# plt.show()