# --> Developed by Marcos Yamasaki
# ------ JSON to Python Dataframe ------
# 10/2020
#
# Description: This script reads a JSON file from nni AutoML and convert it
# to Python pandas Dataframe in order to have a better data visualization.
# Exports it to csv file as well.

import json
import pandas as pd
path2csv = r".\results\ANN"
path = r".\results\ANN\experiment.json"
path2csv = path2csv.replace('\\','/')
path = path.replace('\\','/')
with open(path, "r") as read_file:
    data = json.load(read_file)
    jtopy = json.dumps(data) #json.dumps take a dictionary as input and returns a string as output.
    dict_json = json.loads(jtopy) # json.loads take a string as input and returns a dictionary as output.
    
    for params in dict_json['experimentParameters']['params']['searchSpace']:
        print(params)    
        for values in dict_json['experimentParameters']['params']['searchSpace'][params]['_value']:
            print(values)
            
    i=0
    dataset = pd.DataFrame(columns=['id','r2score','params'])
    for trials in dict_json['trialMessage']:        
        if trials['status'] == 'SUCCEEDED':
            try:
#                print(trials['finalMetricData'][0]['data'])
#                print(trials['id'])
                dataset = dataset.append({
                                'id': trials['id'],
                                'r2score': trials['finalMetricData'][0]['data'],
                                'params': trials['hyperParameters'][0],
                                'duration (min)': (trials['endTime']-trials['startTime'])/1000/60
                                } , ignore_index=True)
                
    #            print(trials['hyperParameters'])
            except KeyError:
                print('error - skip')

paramsList = []
for params in dataset.params:
    teste = json.loads(params)
    paramsList.append(teste['parameters'])

params = pd.DataFrame.from_dict(paramsList)

dataset = pd.concat([dataset, params], axis=1)
dataset.drop(['params'], axis=1, inplace=True)



print(dataset.sort_values(by=['r2score'],ascending=False).head(50))


print('Export dataframe to csv in this path: ')
print(path2csv)
dataset.to_csv(path2csv+'/results.csv', index=None, header=True)




