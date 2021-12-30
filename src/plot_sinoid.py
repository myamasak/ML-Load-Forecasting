# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:21:20 2021

@author: marko
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.close("all")

sns.set(rc={'figure.figsize': (14, 6)})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')


x1 = np.arange(0, 2*np.pi*4, 0.05)
sinx1 = np.sin(x1*1) * 4
sinx2 = np.sin(x1*4) * 4
sinx3 = np.sin(x1*5) * 5
sinx4 = np.sin(x1*2) * 4
#cosx1 = np.cos(x1*3) * 4
plt.figure()
plt.plot(sinx1)
#plt.plot(cosx1)
plt.plot(sinx2)
plt.plot(sinx3)
plt.plot(sinx4)
plt.show()

sinx0 = sinx1 + sinx2 + sinx3 * sinx4

#sinx0 = sinx1 + sinx2 + sinx3

maxNum = -999
minNum = 999

extremas = []
eindex = []

minimas = []
mindex = []

# find extremas first
for i in range(len(sinx0)):
    if sinx0[i] > maxNum and sinx0[i] > 0:
        maxNum = sinx0[i]
    elif sinx0[i] < minNum and sinx0[i] < 0:
        minNum = sinx0[i]
        
    if sinx0[i] > 0:
        if minNum < 0:
            minimas.append(minNum)
            mindex.append(i)
        minNum = 999
    elif sinx0[i] == 0:
        pass
    else:        
        if maxNum > 0:
            extremas.append(maxNum)
            eindex.append(i)
        maxNum = -999
    
#    print(sinx0[i])
    
    

# indices for extremas
N1 = [i for i in range(len(sinx0)) if sinx0[i] in extremas]
N2 = [i for i in range(len(sinx0)) if sinx0[i] in minimas]

from scipy.interpolate import interp1d
f = interp1d(N1, extremas, fill_value="extrapolate")
f2 = interp1d(N1, extremas, kind='cubic', fill_value="extrapolate")
xnew = np.arange(0, N1[-1]+1)
xnew = np.linspace(0, N1[-1], num=100, endpoint=True)

means = []
for i in range(len(extremas)):
    try:
        means.append((extremas[i] + minimas[i]) / 2)
    except IndexError:
        means.append(0)
        
        
f2 = interp1d(N1, means, kind='cubic', fill_value="extrapolate")

#plt.figure()
#plt.plot(xnew, f(xnew))
#plt.plot(xnew, f2(xnew))
#plt.plot(N1, extremas)

plt.figure()
FONTSIZE = 18
plt.plot(sinx0, color=sns_c[0], label='time series')
# plt.plot(xnew, f2(xnew), color=sns_c[2])
plt.plot(N1, extremas, color=sns_c[2], label='upper envelope')
plt.plot(N2, minimas, color=sns_c[3], label='lower envelope')
plt.scatter(N1, extremas, marker=".", s=100, color=sns_c[2], label='extremas')
plt.scatter(N2, minimas, marker=".", s=100, color=sns_c[3], label='minimas')
plt.plot(N1, means, color=sns_c[6], linewidth=1.5, label='mean envelope')
plt.xlabel('Steps', fontsize=FONTSIZE)
plt.ylabel('Amplitude', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
#plt.plot(xnew, f2(xnew), color=sns_c[5])
#plt.plot(pd.DataFrame(means).expanding().mean(), color=sns_c[5])
# plt.plot(pd.DataFrame(sinx0).ewm(alpha=0.01, adjust=False).mean(), color=sns_c[4])

# plt.plot(sinx1, color=sns_c[4])
plt.legend(fontsize=FONTSIZE)
plt.tight_layout()
plt.show()

#
#plt.plot(sinx2)
#plt.plot(sinx3)
#plt.plot(sinx4)


# test = np.where(sinx0 == extremas[0])
# extremas.index(sinx0[1])
