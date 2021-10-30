
### Leituras LightGBM
- https://sefiks.com/2018/10/13/a-gentle-introduction-to-lightgbm-for-applied-machine-learning/
- http://dylan-chen.com/model/lightgbm-tutorial/
- https://lightgbm.readthedocs.io/en/latest/Python-API.html

### Feature importance
- https://stackoverflow.com/questions/53413701/feature-importance-using-lightgbm
- https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm


### Tools
- https://docs.python.org/2/library/sqlite3.html

### nni - localhost
- http://127.0.0.1:8080/detail

### nni docs
- https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html?highlight=nnictl#trial
- https://github.com/microsoft/nni

### conda code sheet
- https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

### Exemplo dissertacao - CRISTIAN SIMIONI MILANI
- https://www.ppgia.pucpr.br/pt/arquivos/mestrado/dissertacoes/2016/Dissertacao_CristianMilani.pdf

#

## Gradient Boosting
- https://en.wikipedia.org/wiki/Gradient_boosting
- https://en.wikipedia.org/wiki/Boosting_(machine_learning)
- >Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.


## XGBoost author - Tianqi Chen 
- https://scholar.google.com/citations?user=7nlvOMQAAAAJ&hl=en
- https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
- The best source of information on XGBoost is the [official GitHub repository for the project](https://github.com/dmlc/xgboost).
- A great source of links with example code and help is [the Awesome XGBoost page](https://github.com/dmlc/xgboost/tree/master/demo).

### Advantages of XGBoost
- https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
- Regularization
- Parallel Processing
- High Flexibility
- Handling Missing Values
- Tree Pruning
- Built-in Cross-Validation
- Continue on Existing Model

## XGBoost - In practice
- https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
- ![kfoldTimeSeries](https://hub.packtpub.com/wp-content/uploads/2019/05/Blocking-Time-Series-Split.png)
- 




### AdaBoost
- >AdaBoost is best used to boost the performance of decision trees on binary classification problems.
- https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/


### Python to Jupyter
- ipynb-py-convert 0.4.5
- https://pypi.org/project/ipynb-py-convert/#description

### Time Series data visualization
- https://machinelearningmastery.com/time-series-data-visualization-with-python/
- https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

### Deep learning time series forecasting
- https://github.com/Alro10/deep-learning-time-series

### Dataset US
- https://www.eia.gov/opendata/bulkfiles.php
- https://www.eia.gov/realtime_grid/#/status?end=20200514T11
#

## NNI - commands
```
nnictl experiment export [experiment_id] --filename [file_path] --type json
nnictl experiment export s08bPjXQ --filename C:\Users\marko\code\ML-Load-Forecasting\results --type csv

nnictl update trialnum --id s08bPjXQ --value 865

nnictl experiment export s08bPjXQ --filename C:\Users\marko\code\ML-Load-Forecasting\results\trial_s08bPjXQ --type json

```

---






##
Sugiro ler partes das seguintes teses (pode te ajudar na escrita de objetivos e contribuições):
https://dr.ntu.edu.sg/bitstream/10356/136779/2/Thesis.pdf

https://ses.library.usyd.edu.au/bitstream/handle/2123/21248/zheng_zw_thesis.pdf?sequence=2&isAllowed=y

---
## 19/10/2021

O que na minha visão da pra melhorar:

- 2 - Revisão da literatura
    - [ ] Adicionar o estado-da-arte para os trabalhos que contém Decomposição + Load forecasting, com os devidos métodos de avaliação de performance (MAPE, R², etc.)
    - [ ] Explicar melhor os conceitos básicos de forecasting time series + machine learning
    - [ ] Explicar sobre outros métodos de decomposição (EWT, EEMD, CEEMDAN, etc.)
- 4 - Machine learning methods
    - [ ] Melhorar a explicação sobre XGBoosting
    - [ ] Melhorar a explicação sobre Gradient boosting
    - [ ] Adicionar Extra Trees, KNN e SVR (talvez não, caso não use)
- 5 - Resultados
    - [ ] Mais texto explicando os detalhes de implementação e resultados
    - [ ] Mais gráficos dos testes
    - [ ] Testes statisticos
    - [ ] E adicionar resultados do 2o dataset (ONS)

-----
.

## Revisão do Leandro
- [ ] 1.3 refs para cada método? EMD, VMD, EEMD, etc.
- [ ] 1.6 incluir uma figura ilustrativa com o fluxo



