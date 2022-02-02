import os
import sys
import glob
path = os.path.realpath(__file__)
path = r'%s' % path.replace(
    f'\\{os.path.basename(__file__)}', '').replace('\\', '/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML', '')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src', '')


DATASET_NAME = 'isone'

#path = path + '/dataset/{DATASET_NAME}/custom/*.csv'

files = glob.glob(path + f'/datasets/{DATASET_NAME}/custom/*.csv')

# files = os.listdir(path)

for index, file in enumerate(files):
    if file.find('forecast7') != -1:
        print(file)
        print('Renaming file...')
    
        filenew = file.replace('_forecast7', '')
        print(filenew)
        os.rename(file, filenew)
#        break
