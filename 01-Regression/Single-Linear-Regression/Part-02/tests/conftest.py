import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope='session')
def single_linear_regression_data() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv('my_test_data/my_test_data.csv')
    yield {
        'response_var': np.array(df['response_var']),
        'independent_var': np.array(df['independent_var'])
    }
    return print('single_linear_regression_data fixture finished.')
