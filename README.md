**This is a Machine Learning for Load Forecasting repository.**

I'm working on Load Forecasting techniques, and this repository has the algorithms and the datasets that will be used in my study.

The datasets are from:
- ISO New England (https://www.iso-ne.com/)
- In the folder /datasets/, there are datasets from 2009-2017. I'll add until 2019.

I'm thinking to add this one as well:
- ONS from Brazil (http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/curva_carga_horaria.aspx)

It's a initial working, so I'm also learning and trying to get some results. Feel free to use.

The techniques that are and will be explored are:
- Artificial Neural Networks;
- Long Short-Term Memory;
- Automated Machine Learning (nni - https://github.com/microsoft/nni);
- Others.

Use these files:
* **ann_impl.py** for ANN
* **LSTM_regression.py** for LSTM


**First results**
```
ANN
3 layers - 8-16-16-1
Removed outliers
Batch_size = 5

Epoch 197/200
58406/58406 [==============================] - 10s 176us/step - loss: 937400.3613
Epoch 198/200
58406/58406 [==============================] - 10s 174us/step - loss: 938468.1414
Epoch 199/200
58406/58406 [==============================] - 10s 172us/step - loss: 937867.7657
Epoch 200/200
58406/58406 [==============================] - 10s 168us/step - loss: 936137.0999
C:\Users\z003t8hn\AppData\Local\Continuum\anaconda3\envs\venv\lib\site-packages\pandas\plotting\_converter.py:129: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.

To register the converters:
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
  warnings.warn(msg, FutureWarning)
The R2 score on the Train set is:       0.884
The R2 score on the Test set is:        0.847

```
![ANN_01](/results/ANN_2009-2017_zoomout_outliers12.png)
![ANN_02](/results/ANN_2009-2017_zoomout_outliers12_2.png)
![ANN_03](/results/ANN_2009-2017_zoomin_outliers12.png)


-------

**Best trial** (until 22/08)

```
22/08/2019
ANN
4 layers - 64-64-64-64-1
Removed outliers
Batch_size = 20
Temperature + Date only

Epoch 96/100
58406/58406 [==============================] - 2s 42us/step - loss: 761372.1349
Epoch 97/100
58406/58406 [==============================] - 2s 41us/step - loss: 763797.9499
Epoch 98/100
58406/58406 [==============================] - 3s 44us/step - loss: 765644.7697
Epoch 99/100
58406/58406 [==============================] - 2s 41us/step - loss: 764428.3648
Epoch 100/100
58406/58406 [==============================] - 2s 41us/step - loss: 763874.4192
C:\Users\z003t8hn\AppData\Local\Continuum\anaconda3\envs\venv\lib\site-packages\pandas\plotting\_converter.py:129: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.

To register the converters:
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
  warnings.warn(msg, FutureWarning)
The R2 score on the Train set is:       0.910
The R2 score on the Test set is:        0.877
```

![ANN_220819_01](/results/ANN_2009-2017_64_4layers.png)
![ANN_220819_02](/results/ANN_2009-2017_64_4layers_2.png)
![ANN_220819_03](/results/ANN_2009-2017_64_4layers_3.png)
![ANN_220819_04](/results/ANN_2009-2017_64_4layers_4.png)




----------------
**UPDATE**

**NNI - Neural Network Intelligence**
(11/09/19)

![ANN_110919_01](/results/ANN_2009-2017_NNI_110919.PNG)
![ANN_110919_02](/results/ANN_2009-2017_NNI_110919_02.PNG)
![ANN_110919_03](/results/ANN_2009-2017_NNI_110919_03.PNG)
