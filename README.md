# Basic-ML-OOP  

Scott Stoltzman  


As data scientists, many of us are used to procedural programming. This repo will correspond to a set of blog posts on my blog -> [Stoltzmaniac](https://stoltzmaniac.com).  

The goal is to make a basic library of machine learning objects in order to help us solve problems in an object oriented fashion. each part will live within its own subdirectory to keep it all contained. Each part will have its own `requirements.txt` so you must run your code from those directories to work.  


In order to avoid flooding this README and wind up repeating my workload, you can checkout my blog: [Stoltzmaniac](https://www.stoltzmaniac.com/]


In order to get yourself going, you have to go into the directory and then start playing with the import statements, etc.
```bash
cd 01-Regression/Linear-Regression/Part-02
python
```

Here is an example of doing linear regression (both single and multivariate) and PCA.

Setup:
```python
import pandas as pd
from models.regression import LinearRegression
from models.dimensionizers import PrincipalComponentAnalysis
from sklearn.datasets import load_iris
iris = load_iris()
```

Single linear regression (we know we can explain 100% of the variance)
```python
df = pd.read_csv('my_test_data/my_test_data.csv')
data_x = df[['x']]
data_y = df['y']

single_linear_regression = LinearRegression(predictor_vars=data_x,
                                            response_var=data_y,
                                            train_split=0.7,
                                            seed=123,
                                            scale_type='normalize',
                                            learning_rate=0.01,
                                            tolerance=0.00001,
                                            batch_size=12,
                                            max_epochs=20000,
                                            decay = 0.90)
single_linear_regression.fit_stochastic_gradient_descent()
print(single_linear_regression)
```
Results
```bash
Epoch: 0 - Error: 27144.581707246718
Epoch: 100 - Error: 1609.1757946720165
Epoch: 200 - Error: 320.2291601027205
Epoch: 300 - Error: 63.72092034669193
Epoch: 400 - Error: 12.681006265402576
Epoch: 500 - Error: 2.5242284229174152
Epoch: 600 - Error: 0.5027575170076343
Epoch: 700 - Error: 0.10010355028842773
Epoch: 800 - Error: 0.019927613049417967
Epoch: 900 - Error: 0.003966926052434834
Epoch: 1000 - Error: 0.0007910657321511311
Epoch: 1100 - Error: 0.0006941726959010522
Epoch: 1200 - Error: 0.0006941702877259255
Epoch: 1300 - Error: 0.000694170287690926
Converged
            Model Results
            -------------
            Betas: [(0, 151.27102352227334), (1, 303.9119753361067)]
            R^2 Train: 0.9999999161580064
            R^2 Test: 0.9999999129705581

```



Multivariate linear regression (Cannot explain 100% of the variance)
```python
df = pd.read_csv('my_test_data/my_test_data_2.csv')
data_x = df[['i1', 'i2', 'i3']]
data_y = df['d']

multiple_linear_regression = LinearRegression(predictor_vars=data_x,
                                            response_var=data_y,
                                            train_split=0.7,
                                            seed=123,
                                            learning_rate=0.01,
                                            scale_type='normalize',
                                            tolerance=0.00001,
                                            batch_size=12,
                                            max_epochs=20000,
                                            decay = 0.90)
multiple_linear_regression.fit_stochastic_gradient_descent()
print(multiple_linear_regression)
```

Results
```bash
Epoch: 0 - Error: 7.784441767333225
Epoch: 100 - Error: 0.4958346178634603
Epoch: 200 - Error: 0.2430738431764785
Epoch: 300 - Error: 0.16598546640597142
Epoch: 400 - Error: 0.12297075579916167
Epoch: 500 - Error: 0.09722505204032147
Epoch: 600 - Error: 0.0820194181675479
Epoch: 700 - Error: 0.07296038200103765
Epoch: 800 - Error: 0.06711456097026484
Epoch: 900 - Error: 0.06334242045385301
Epoch: 1000 - Error: 0.06092582633687668
Epoch: 1100 - Error: 0.060768212397990545
Epoch: 1200 - Error: 0.06076820846259531
Epoch: 1300 - Error: 0.060768208462492534
Epoch: 1400 - Error: 0.060768208462492534
Epoch: 1500 - Error: 0.06076820846249252
Epoch: 1600 - Error: 0.060768208462492534
Epoch: 1700 - Error: 0.06076820846249252
Epoch: 1800 - Error: 0.060768208462492534
Epoch: 1900 - Error: 0.060768208462492534
Epoch: 2000 - Error: 0.06076820846249252
Epoch: 2100 - Error: 0.060768208462492534
Epoch: 2200 - Error: 0.060768208462492534
Epoch: 2300 - Error: 0.06076820846249252
Epoch: 2400 - Error: 0.060768208462492534
Epoch: 2500 - Error: 0.06076820846249252
Epoch: 2600 - Error: 0.060768208462492534
Epoch: 2700 - Error: 0.060768208462492534
Epoch: 2800 - Error: 0.060768208462492534
Epoch: 2900 - Error: 0.060768208462492534
Epoch: 3000 - Error: 0.06076820846249252
Epoch: 3100 - Error: 0.060768208462492534
Epoch: 3200 - Error: 0.060768208462492534
Epoch: 3300 - Error: 0.060768208462492534
Epoch: 3400 - Error: 0.06076820846249252
Epoch: 3500 - Error: 0.06076820846249252
Epoch: 3600 - Error: 0.060768208462492534
Epoch: 3700 - Error: 0.060768208462492534
Epoch: 3800 - Error: 0.060768208462492534
Epoch: 3900 - Error: 0.060768208462492534
Epoch: 4000 - Error: 0.060768208462492534
Epoch: 4100 - Error: 0.060768208462492534
Epoch: 4200 - Error: 0.060768208462492534
Epoch: 4300 - Error: 0.06076820846249252
Epoch: 4400 - Error: 0.060768208462492534
Epoch: 4500 - Error: 0.06076820846249252
Epoch: 4600 - Error: 0.06076820846249252
Epoch: 4700 - Error: 0.060768208462492534
Epoch: 4800 - Error: 0.060768208462492534
Epoch: 4900 - Error: 0.060768208462492534
Epoch: 5000 - Error: 0.06076820846249252
Epoch: 5100 - Error: 0.06076820846249252
Epoch: 5200 - Error: 0.06076820846249252
Epoch: 5300 - Error: 0.06076820846249252
Epoch: 5400 - Error: 0.06076820846249252
Epoch: 5500 - Error: 0.060768208462492534
Epoch: 5600 - Error: 0.06076820846249252
Epoch: 5700 - Error: 0.060768208462492534
Epoch: 5800 - Error: 0.060768208462492534
Epoch: 5900 - Error: 0.06076820846249252
Epoch: 6000 - Error: 0.060768208462492534
Epoch: 6100 - Error: 0.06076820846249252
Epoch: 6200 - Error: 0.060768208462492534
Epoch: 6300 - Error: 0.060768208462492534
Epoch: 6400 - Error: 0.06076820846249252
Epoch: 6500 - Error: 0.060768208462492534
Epoch: 6600 - Error: 0.06076820846249252
Epoch: 6700 - Error: 0.06076820846249252
Epoch: 6800 - Error: 0.06076820846249252
Epoch: 6900 - Error: 0.060768208462492534
Epoch: 7000 - Error: 0.06076820846249252
Epoch: 7100 - Error: 0.060768208462492534
Epoch: 7200 - Error: 0.060768208462492534
Epoch: 7300 - Error: 0.060768208462492534
Epoch: 7400 - Error: 0.060768208462492534
Epoch: 7500 - Error: 0.06076820846249252
Epoch: 7600 - Error: 0.06076820846249252
Epoch: 7700 - Error: 0.060768208462492534
Epoch: 7800 - Error: 0.060768208462492534
Epoch: 7900 - Error: 0.060768208462492534
Epoch: 8000 - Error: 0.060768208462492534
Epoch: 8100 - Error: 0.06076820846249252
Epoch: 8200 - Error: 0.06076820846249252
Epoch: 8300 - Error: 0.06076820846249252
Epoch: 8400 - Error: 0.060768208462492534
Epoch: 8500 - Error: 0.06076820846249252
Epoch: 8600 - Error: 0.060768208462492534
Epoch: 8700 - Error: 0.060768208462492534
Epoch: 8800 - Error: 0.060768208462492534
Epoch: 8900 - Error: 0.06076820846249252
Epoch: 9000 - Error: 0.06076820846249252
Epoch: 9100 - Error: 0.060768208462492534
Epoch: 9200 - Error: 0.060768208462492534
Epoch: 9300 - Error: 0.06076820846249252
Epoch: 9400 - Error: 0.06076820846249252
Epoch: 9500 - Error: 0.060768208462492534
Epoch: 9600 - Error: 0.060768208462492534
Epoch: 9700 - Error: 0.060768208462492534
Epoch: 9800 - Error: 0.060768208462492534
Epoch: 9900 - Error: 0.06076820846249252
Epoch: 10000 - Error: 0.06076820846249252
Epoch: 10100 - Error: 0.06076820846249251
Epoch: 10200 - Error: 0.060768208462492534
Epoch: 10300 - Error: 0.06076820846249252
Epoch: 10400 - Error: 0.06076820846249252
Epoch: 10500 - Error: 0.06076820846249252
Epoch: 10600 - Error: 0.06076820846249252
Epoch: 10700 - Error: 0.06076820846249252
Epoch: 10800 - Error: 0.060768208462492534
Epoch: 10900 - Error: 0.060768208462492534
Epoch: 11000 - Error: 0.060768208462492534
Epoch: 11100 - Error: 0.060768208462492534
Epoch: 11200 - Error: 0.06076820846249252
Epoch: 11300 - Error: 0.06076820846249252
Epoch: 11400 - Error: 0.060768208462492534
Epoch: 11500 - Error: 0.06076820846249252
Epoch: 11600 - Error: 0.06076820846249252
Epoch: 11700 - Error: 0.060768208462492534
Epoch: 11800 - Error: 0.06076820846249252
Epoch: 11900 - Error: 0.060768208462492534
Epoch: 12000 - Error: 0.060768208462492534
Epoch: 12100 - Error: 0.06076820846249252
Epoch: 12200 - Error: 0.060768208462492534
Epoch: 12300 - Error: 0.06076820846249252
Epoch: 12400 - Error: 0.06076820846249252
Epoch: 12500 - Error: 0.06076820846249252
Epoch: 12600 - Error: 0.06076820846249252
Epoch: 12700 - Error: 0.060768208462492534
Epoch: 12800 - Error: 0.06076820846249252
Epoch: 12900 - Error: 0.06076820846249252
Epoch: 13000 - Error: 0.060768208462492534
Epoch: 13100 - Error: 0.060768208462492534
Epoch: 13200 - Error: 0.06076820846249252
Epoch: 13300 - Error: 0.06076820846249252
Epoch: 13400 - Error: 0.06076820846249252
Epoch: 13500 - Error: 0.060768208462492534
Epoch: 13600 - Error: 0.060768208462492534
Epoch: 13700 - Error: 0.06076820846249252
Epoch: 13800 - Error: 0.060768208462492534
Epoch: 13900 - Error: 0.06076820846249252
Epoch: 14000 - Error: 0.060768208462492534
Epoch: 14100 - Error: 0.06076820846249252
Epoch: 14200 - Error: 0.060768208462492534
Epoch: 14300 - Error: 0.06076820846249252
Epoch: 14400 - Error: 0.060768208462492534
Epoch: 14500 - Error: 0.06076820846249252
Epoch: 14600 - Error: 0.06076820846249252
Epoch: 14700 - Error: 0.06076820846249252
Epoch: 14800 - Error: 0.06076820846249252
Epoch: 14900 - Error: 0.06076820846249252
Epoch: 15000 - Error: 0.060768208462492534
Epoch: 15100 - Error: 0.06076820846249252
Epoch: 15200 - Error: 0.060768208462492534
Epoch: 15300 - Error: 0.06076820846249252
Epoch: 15400 - Error: 0.060768208462492534
Epoch: 15500 - Error: 0.060768208462492534
Epoch: 15600 - Error: 0.06076820846249252
Epoch: 15700 - Error: 0.06076820846249252
Epoch: 15800 - Error: 0.06076820846249252
Epoch: 15900 - Error: 0.06076820846249252
Epoch: 16000 - Error: 0.060768208462492534
Epoch: 16100 - Error: 0.06076820846249252
Epoch: 16200 - Error: 0.060768208462492534
Epoch: 16300 - Error: 0.06076820846249252
Epoch: 16400 - Error: 0.06076820846249252
Epoch: 16500 - Error: 0.060768208462492534
Epoch: 16600 - Error: 0.060768208462492534
Epoch: 16700 - Error: 0.06076820846249252
Epoch: 16800 - Error: 0.06076820846249252
Epoch: 16900 - Error: 0.060768208462492534
Epoch: 17000 - Error: 0.06076820846249252
Epoch: 17100 - Error: 0.060768208462492534
Epoch: 17200 - Error: 0.06076820846249252
Epoch: 17300 - Error: 0.060768208462492534
Epoch: 17400 - Error: 0.06076820846249252
Epoch: 17500 - Error: 0.060768208462492534
Epoch: 17600 - Error: 0.06076820846249252
Epoch: 17700 - Error: 0.06076820846249252
Epoch: 17800 - Error: 0.060768208462492534
Epoch: 17900 - Error: 0.06076820846249252
Epoch: 18000 - Error: 0.060768208462492534
Epoch: 18100 - Error: 0.060768208462492534
Epoch: 18200 - Error: 0.06076820846249252
Epoch: 18300 - Error: 0.060768208462492534
Epoch: 18400 - Error: 0.06076820846249252
Epoch: 18500 - Error: 0.060768208462492534
Epoch: 18600 - Error: 0.060768208462492534
Epoch: 18700 - Error: 0.06076820846249252
Epoch: 18800 - Error: 0.060768208462492534
Epoch: 18900 - Error: 0.06076820846249252
Epoch: 19000 - Error: 0.060768208462492534
Epoch: 19100 - Error: 0.060768208462492534
Epoch: 19200 - Error: 0.060768208462492534
Epoch: 19300 - Error: 0.060768208462492534
Epoch: 19400 - Error: 0.060768208462492534
Epoch: 19500 - Error: 0.06076820846249252
Epoch: 19600 - Error: 0.060768208462492534
Epoch: 19700 - Error: 0.06076820846249252
Epoch: 19800 - Error: 0.06076820846249252
Epoch: 19900 - Error: 0.060768208462492534
Epoch: 20000 - Error: 0.060768208462492534
Max Epochs limit reached
            Model Results
            -------------
            Betas: [(0, 1.0001875941305407), (1, 0.11089304007782451), (2, 0.35644258684939834), (3, 1.0038468821153166)]
            R^2 Train: 0.6239899516410836
            R^2 Test: 0.42878692987237976
```


Principal Component Analysis (PCA)
```python
data_x = iris['data']
data_y = iris['target']
pca = PrincipalComponentAnalysis(
    predictor_vars=data_x,
    response_var=data_y,
    scale_type='normalize',
    train_split=0.7,
    seed=123,
    variance_explained_cutoff=0.95
)
print(pca)
```

Results
```bash

Variance Explained Cutoff: 0.95
PCA Variance Explained: [0.84141262]
Eigenvalues: 
[0.24742686]
Eigenvectors: 
[[ 0.43061003]
[-0.15830383]
[ 0.61817815]
[ 0.63825597]]
```