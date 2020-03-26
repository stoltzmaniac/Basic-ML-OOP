import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


@pytest.fixture(scope="session")
def single_linear_regression_data() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv("my_test_data/my_test_data.csv")
    yield {
        "response_var": df["y"],
        "predictor_vars": df["x"],
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
        "response_var": df["d"],
        "predictor_vars": df[["i1", "i2", "i3"]],
    }
    return print("multiple_linear_regression_data fixture finished.")


@pytest.fixture(scope="session")
def iris_data() -> dict:
    """
    Setup test data for
    :return:
    """
    df = load_iris()
    yield {
        "response_var": np.array(df)[:, -1],
        "predictor_vars": np.array(df)[:, :3],
    }
    return print("multiple_linear_regression_data fixture finished.")
