**This is a Machine Learning for Load Forecasting repository.**

Focused on Load Forecasting techniques, this repository has the algorithms and the datasets that have been used in my study.

The paper has been published, you can check here:
https://www.sciencedirect.com/science/article/pii/S0142061523006361

The datasets are from:
- ISO New England (https://www.iso-ne.com/). In the folder datasets/, there are datasets from 2003-2019.
- ONS from Brazil (http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/curva_carga_horaria.aspx)

This work is done, no further updates is aimed for this repo. Let me know if you have any questions.

The Forecasting techniques used are:
- Combination of GBR, XGBoost, KNN, and SVR with decomposition techniques STL, EMD, EEMD, CEEMDAN, and EWT.
- Automated Machine Learning (nni - Neural Network Intelligence https://github.com/microsoft/nni);

Main file:
* **src/TimeSeriesDecompose.py** - this is the main source file that you'll need to execute.
* autoML/config_TimeSeriesDecompose.yml - the yml files are for NNI usage, configure them as needed (not mandatory).



