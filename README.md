# Basic-ML-OOP  

Scott Stoltzman  


As data scientists, many of us are used to procedural programming. This repo will correspond to a set of blog posts on my blog -> [Stoltzmaniac](https://stoltzmaniac.com).  

The goal is to make a basic library of machine learning objects in order to help us solve problems in an object oriented fashion. each part will live within its own subdirectory to keep it all contained. Each part will have its own `requirements.txt` so you must run your code from those directories to work.  


In order to avoid flooding this README and wind up repeating my workload, you can checkout my blog: [Stoltzmaniac](https://www.stoltzmaniac.com/]


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