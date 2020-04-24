import json
import os
import numpy as np
import pandas as pd

path = r'%s' % os.getcwd().replace('\\','/')
path += r'/results/trial_s08bPjXQ.json'

dataList=[]

with open(path) as json_file:
    data = json.load(json_file)
     
newlist = sorted(data, key=lambda k: k['value'], reverse=True) 


#print(json.dumps(data, indent = 4, sort_keys=True))

df = pd.DataFrame(newlist)

newlist2 = []
for parameter in df.parameter:
    json3 = str(parameter).replace("'",'"')
    newlist2.append(pd.read_json(json3, lines=True))

Parameter = pd.concat(newlist2, axis=0, sort=False, ignore_index=True)
df = df.drop(['parameter'],axis=1)
df = pd.concat([df,Parameter],axis=1)


df.hist("learning_rate")
import seaborn
seaborn.pairplot(df, vars=['colsample_bytree', 'gamma', 'learning_rate'],
                 kind='reg')

sortValues = df.sort_values('value', ascending=False).head(50)

sortValues.hist()

for i in range(0,len(sortValues.columns)):
    if (sortValues.columns[i].find("value") == -1
        and sortValues.columns[i].find("id") == -1):    
        print(str(sortValues.columns[i]) + "=" + str(pd.value_counts(sortValues[str(sortValues.columns[i])].values, sort=True).nlargest(1)).split()[0])
        
        


