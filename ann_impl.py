# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:02:14 2019

@author: z003t8hn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:04:58 2019

@author: z003t8hn
"""
import numpy as np
import pandas as pd
import os
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Importing the dataset
dataset17 = pd.read_excel(os.getcwd() + '/datasets/2017_smd_hourly.xls', 'ISO NE CA')
dataset16 = pd.read_excel(os.getcwd() + '/datasets/2016_smd_hourly.xls', 'ISO NE CA')
dataset15 = pd.read_excel(os.getcwd() + '/datasets/2015_smd_hourly.xls', 'ISONE CA')
dataset14 = pd.read_excel(os.getcwd() + '/datasets/2014_smd_hourly.xls', 'ISONE CA')
dataset13 = pd.read_excel(os.getcwd() + '/datasets/2013_smd_hourly.xls', 'ISONE CA')
dataset12 = pd.read_excel(os.getcwd() + '/datasets/2012_smd_hourly.xls', 'ISONE CA')
dataset11 = pd.read_excel(os.getcwd() + '/datasets/2011_smd_hourly.xls', 'ISONE CA')
dataset10 = pd.read_excel(os.getcwd() + '/datasets/2010_smd_hourly.xls', 'ISONE CA')
dataset09 = pd.read_excel(os.getcwd() + '/datasets/2009_smd_hourly.xls', 'ISONE CA')

concatlist = [dataset09,dataset10,dataset11,dataset12,dataset13,dataset14,dataset15,dataset16,dataset17]
dataset = pd.concat(concatlist,axis=0,sort=False,ignore_index=True)

X = dataset.iloc[:, :]
X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','Date','Hour','RT_LMP','RT_EC','RT_CC','RT_MLC','SYSLoad','RegSP','RegCP'], axis=1)
#X = X.drop(['DEMAND','DA_DEMD','DA_LMP','DA_EC','DA_CC','DA_MLC','RT_LMP','RT_EC','RT_CC','RT_MLC'], axis=1)
y = dataset.iloc[:, 3]


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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainsc = sc.fit_transform(X_train)
X_testsc = sc.transform(X_test)

#Usually it's a good practice to apply following formula in order to find out
#the total number of hidden layers needed.
#Nh = Ns/(α∗ (Ni + No))
#where
#
#Ni = number of input neurons.
#No = number of output neurons.
#Ns = number of samples in training data set.
#α = an arbitrary scaling factor usually 2-10.

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(16, activation = 'relu', input_dim = 6))

# Adding the second hidden layer
model.add(Dense(units = 16, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 16, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_trainsc, y_train, batch_size = 10, epochs = 200)


rows = X_test.index
df2 = df.iloc[rows[0]:]

y_pred = model.predict(X_testsc)
y_tested = y_test.reset_index()
y_tested = y_tested.drop(['index'],axis=1)
plt.figure(1)
#plt.plot(df2,y_tested, color = 'red', label = 'Real data')
plt.plot(df,y, color = 'gray', label = 'Real data')
plt.plot(df2,y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
y_pred_train = model.predict(X_trainsc)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))



#A = pd.DataFrame(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
#b = pd.DataFrame(np.array([11, 12, 13, 14, 15, 16, 17, 18]))
#A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2, random_state = 0, shuffle = False)

