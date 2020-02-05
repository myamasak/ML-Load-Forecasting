# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:26:28 2020

@author: z003t8hn
"""

import numpy as np
#import pandas as pd

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
import modin.pandas as pd


