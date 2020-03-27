import numpy as np
import pytest
from models.data_handler import PreProcessData


@pytest.fixture(scope="module")
def pre_process_data_dataframe_single(single_linear_regression_data):
    input_data = PreProcessData(
        predictor_vars=single_linear_regression_data['predictor_vars'],
        response_var=single_linear_regression_data['response_var'],
        train_split=0.7,
        seed=123,
        scale_type='normalize'
    )
    print(input_data)
    return input_data


@pytest.fixture(scope="module")
def pre_process_data_dataframe_multiple(multiple_linear_regression_data):
    input_data = PreProcessData(
        predictor_vars=multiple_linear_regression_data['predictor_vars'],
        response_var=multiple_linear_regression_data['response_var'],
        train_split=0.7,
        seed=123,
        scale_type='normalize'
    )
    print(input_data)
    return input_data


def test_pre_process_data_dataframe_single(pre_process_data_dataframe_single, single_linear_regression_data):
    assert type(pre_process_data_dataframe_single.response_var) == np.ndarray
    assert type(pre_process_data_dataframe_single.predictor_vars) == np.ndarray
    assert all(pre_process_data_dataframe_single.predictor_vars) == all(single_linear_regression_data['predictor_vars'])
    assert all(pre_process_data_dataframe_single.response_var) == all(single_linear_regression_data['response_var'])

def test_pre_process_data_dataframe_multiple(pre_process_data_dataframe_multiple, multiple_linear_regression_data):
    assert type(pre_process_data_dataframe_multiple.response_var) == np.ndarray
    assert type(pre_process_data_dataframe_multiple.predictor_vars) == np.ndarray
    assert all(pre_process_data_dataframe_multiple.response_var) == all(multiple_linear_regression_data['response_var'])
    # TODO: fix multiple array of array tests
    # assert all(pre_process_data_dataframe_multiple.predictor_vars) == all(multiple_linear_regression_data['predictor_vars'])
