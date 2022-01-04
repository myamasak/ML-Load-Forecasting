
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


-----

## Próximos passos 03/01/2021

1) Usar uma CV diferente dos 40-fold. É mais comum 10-fold. Leia nos aritigos
que usam o dataset o que usam. Seria possível comparar resultados com eles?
- Sim, vou verificar se é possível realizar uma comparação. Existem vários artigos da competição Gefcom (Global Energy Forecasting Competition), porém acredito que eles utilizaram outras partes do dataset (ao invés do nível hierárquico mais alto, como eu utilizei) e para as versões mais recentes da competição (a última foi 2017), utilizaram-se de técnicas diferentes de probabilidade de previsão com nível hierárquico, mas imagino que dê para comparar alguns resultados sim se adaptar minha implementação com os mesmos dados de dataset considerados.

2) Remover o R2 e usarmos menos métricas. Leia o artigo do Kaggle do email anterior.
Ok!


### Dos comentários anotei as seguintes sugestões:

## Comentários da banca:

 

- [ ] Remover R² das métricas de avaliação dos modelos. A janela deslizante muda a média no qual deteriora o R² (resultados negativos)
- [ ] Descrever nas tabelas de resultados o periodo de validação e treinamento
- [ ] Verificar os títulos corretos para tabelas/figuras, se baseando nos artigos publicados
- [ ] Corrigir nas tabelas de resultado o período de validação
- [ ] Descrever no texto o 40-fold Cross-validation ao invés de colocar nas tabelas
- [ ] Incluir os custos computacionais e tempo de cada modelo treinado
- [ ] Incluir os custos computacionais de cada método de decomposição
- [ ] Colocar a justificativa para o número de IMFs para cada método de decomposição
- [ ] Melhorar a figura do modelo de framework colocando apenas os modelos utilizados
- [ ] Colocar os resultados dos modelos utilizados KNN, SVR, LSTM e seus custos computacionais
- [ ] Descrever melhor no documento os modelos citados no documento (KNN, SVR)
- [ ] Aumentar o período de validação para cobrir melhor o dataset (pegar todas sazonalidades) e talvez diminuir o número de folds
- [ ] Descrever com clareza de quais regiões foram utilizadas dos datasets
- [ ] Descrever quais dados e quais matrizes energéticas foram utilizados dos datasets
- [ ] Justificativas para não-utilização/utilização de ensemble de KNN SVR LSTM GBR XGBOOST, e seus custos computacionais
- [ ] Avaliar a quantidade de 40-folds e reduzir para um número mais plausível com a literatura
- [ ] Explicar as diferenças das métricas de erros nos resultados, também para tuning hiperparametros porque RMSE é a melhor métrica para focar (diferenciável) - ler artigo enviado pelo Leandro
- [ ] MIMO e SISO, explicitar no documento as justificativas para cada dataset
- [ ] Justificar o uso do período de 2015-2018 para os 2 datasets (evitar o período de COVID)
 

 

## Próximas etapas

- [ ] Focar nos regressores (atrasos) utilizando PACF
- [ ] Tunagem de hiperparametros para todos os modelos e IMFs
- [ ] Artigo em periódico
 

## Dos slides

- [ ] Aumentar os dados de teste, incluir diferentes dias e estações do ano, incluindo finais de semana.
- [ ] Ensemble of GBR algorithm with more heterogeneous algorithms, such as KNN, SVR, LSTM and more.
- [ ] Statistical tests: Diebold-Mariano (DM).
- [ ] Include Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) on results.

