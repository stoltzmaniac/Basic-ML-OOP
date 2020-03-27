import numpy as np
import pytest
from models.data_handler import SplitTestTrain


@pytest.fixture(scope="module")
def split_test_train_dataframe_single(single_linear_regression_data):
    input_data = SplitTestTrain(
        predictor_vars=single_linear_regression_data['predictor_vars'],
        response_var=single_linear_regression_data['response_var'],
        train_split=0.7,
        seed=123
    )
    print(input_data)
    return input_data


@pytest.fixture(scope="module")
def split_test_train_dataframe_multiple(multiple_linear_regression_data):
    input_data = SplitTestTrain(
        predictor_vars=multiple_linear_regression_data['predictor_vars'],
        response_var=multiple_linear_regression_data['response_var'],
        train_split=0.7,
        seed=123
    )
    print(input_data)
    return input_data


def test_split_test_train_dataframe_single(split_test_train_dataframe_single, single_linear_regression_data):
    assert type(split_test_train_dataframe_single.response_var) == np.ndarray
    assert type(split_test_train_dataframe_single.predictor_vars) == np.ndarray
    assert all(split_test_train_dataframe_single.predictor_vars) == all(single_linear_regression_data['predictor_vars'])
    assert all(split_test_train_dataframe_single.response_var) == all(single_linear_regression_data['response_var'])

def test_split_test_train_dataframe_multiple(split_test_train_dataframe_multiple, multiple_linear_regression_data):
    assert type(split_test_train_dataframe_multiple.response_var) == np.ndarray
    assert type(split_test_train_dataframe_multiple.predictor_vars) == np.ndarray
    assert all(split_test_train_dataframe_multiple.response_var) == all(multiple_linear_regression_data['response_var'])
    # TODO: fix multiple array of array tests
    # assert all(split_test_train_dataframe_multiple.predictor_vars) == all(multiple_linear_regression_data['predictor_vars'])
