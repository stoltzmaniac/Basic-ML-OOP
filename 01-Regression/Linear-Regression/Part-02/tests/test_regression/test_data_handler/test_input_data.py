import numpy as np
import pytest
from models.data_handler import InputData


@pytest.fixture(scope="module")
def input_data_dataframe_single(single_linear_regression_data):
    input_data = InputData(
        predictor_vars=single_linear_regression_data['predictor_vars'],
        response_var=single_linear_regression_data['response_var']
    )
    print(input_data)
    return input_data

@pytest.fixture(scope="module")
def input_data_ndarray_single(single_linear_regression_data):
    input_data = InputData(
        predictor_vars=np.array(single_linear_regression_data['predictor_vars']),
        response_var=np.array(single_linear_regression_data['response_var'])
    )
    print(input_data)
    return input_data

@pytest.fixture(scope="module")
def input_data_dataframe_multiple(multiple_linear_regression_data):
    input_data = InputData(
        predictor_vars=multiple_linear_regression_data['predictor_vars'],
        response_var=multiple_linear_regression_data['response_var']
    )
    print(input_data)
    return input_data

@pytest.fixture(scope="module")
def input_data_ndarray_multiple(multiple_linear_regression_data):
    input_data = InputData(
        predictor_vars=np.array(multiple_linear_regression_data['predictor_vars']),
        response_var=np.array(multiple_linear_regression_data['response_var'])
    )
    print(input_data)
    return input_data



def test_input_data_dataframe_single(input_data_dataframe_single, single_linear_regression_data):
    assert type(input_data_dataframe_single.response_var) == np.ndarray
    assert type(input_data_dataframe_single.predictor_vars) == np.ndarray
    assert all(input_data_dataframe_single.response_var) == all(np.array(single_linear_regression_data['response_var']))
    assert all(input_data_dataframe_single.predictor_vars) == all(np.array(single_linear_regression_data['predictor_vars']))

def test_input_data_dataframe_multiple(input_data_dataframe_multiple, multiple_linear_regression_data):
    assert type(input_data_dataframe_multiple.response_var) == np.ndarray
    assert type(input_data_dataframe_multiple.predictor_vars) == np.ndarray
    assert all(input_data_dataframe_multiple.response_var) == all(np.array(multiple_linear_regression_data['response_var']))
    # TODO: Fix this test, not sure how to compare array of arrays
    #assert all(input_data_dataframe_multiple.predictor_vars) == all(np.array(multiple_linear_regression_data['response_var']))
