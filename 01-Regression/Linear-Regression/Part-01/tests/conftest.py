import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def single_linear_regression_data() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv("my_test_data/my_test_data.csv")
    yield {
        "response_var": np.array(df)[:, -1],
        "predictor_vars": np.array(df)[:, :1],
    }
    return print("single_linear_regression_data fixture finished.")


@pytest.fixture(scope="session")
def multiple_linear_regression_data() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv("my_test_data/my_test_data_2.csv")
    yield {
        "response_var": np.array(df)[:, -1],
        "predictor_vars": np.array(df)[:, :3],
    }
    return print("multiple_linear_regression_data fixture finished.")
