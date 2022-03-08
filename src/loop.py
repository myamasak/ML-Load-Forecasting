# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 22:26:48 2022

@author: marko

This script is aimed to create a loop for TimeSeriesDecompose.py file, using
different args to find the optimal parameters.
"""

import sys
import os
import subprocess
import time
import logging
import pandas as pd
import numpy as np
from log import log
start_time = time.time()

# Set path to import dataset and export figures
path = os.path.realpath(__file__)
# path = r'%s' % path.replace(f'\\{os.path.basename(__file__)}', '')
path = r'%s' % path.replace(
    f'\\{os.path.basename(__file__)}', '').replace('\\', '/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('\\autoML', '\\src')
# elif path.find('src') != -1:
#     path = r'%s' % path.replace('/src', '')

script_path = 'TimeSeriesDecompose.py'
dataset = 'ISONE'
algorithms = ['xgboost', 'gbr', 'svr', 'knn']
# algorithms = ['svr', 'knn'] 
modes = ['emd', 'ewt', 'eemd', 'ceemdan', 'stl-a']
# modes = ['eemd', 'ceemdan']
# modes = ['emd', 'ceemdan']    
nmodes = np.arange(1,10)
pypath = 'C:/Users/marko/Anaconda3/envs/venvAUTO_CPU/python.exe'

for algo in algorithms:
    for mode in modes:
        for nmode in nmodes:
            if mode.find('ewt') != -1 and nmode == 1:
                continue
            if mode.find('stl-a') != -1 and nmode != 1:
                continue
            cmd = [pypath, script_path,
            "-algo", algo, "-mode", mode, "-nmodes", str(nmode), "-dataset", dataset, "-loadoff"
            "-loop"]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            stdout, err = proc.communicate()
            lines = stdout.decode()            
            print(lines)
                

# Close logging handlers to release the log file
handlers = logging.getLogger().handlers[:]
for handler in handlers:
    handler.close()
    logging.getLogger().removeHandler(handler)