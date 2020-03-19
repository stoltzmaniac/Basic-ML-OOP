import numpy as np
import pytest

from regression import LinearRegression


@pytest.fixture(scope='module')
def single_linear_regression_model(single_linear_regression_data):
    linear_regression_model = LinearRegression(independent_vars=single_linear_regression_data['independent_vars'],
                                               dependent_var=single_linear_regression_data['dependent_var'],
                                               iterations=10000,
                                               learning_rate=0.001,
                                               train_split=0.7,
                                               seed=123)
    return linear_regression_model


def test_single_linear_regression_data_passing_correctly(single_linear_regression_model, single_linear_regression_data):
    """
    Setup linear regression model
    :return:
    """
    assert(single_linear_regression_model.independent_vars_train.all() == single_linear_regression_data['independent_vars'].all())
    assert(single_linear_regression_model.dependent_var_train.all() == single_linear_regression_data['dependent_var'].all())
    assert(type(single_linear_regression_model.independent_vars_train) == np.ndarray)
    assert(type(single_linear_regression_model.dependent_var_train) == np.ndarray)

def test_single_linear_regression_coefficients(single_linear_regression_model):
    """
    Test regression model coefficients
    :return:
    """
    print(single_linear_regression_model)
    expected_coefficients = [(0, 1.9976245504100723), (1, 1.9705419778081599)]
    no_of_betas = len(single_linear_regression_model.B)
    for n in range(no_of_betas):
        assert(pytest.approx(expected_coefficients[n][1], 0.001) == single_linear_regression_model.B[n])

def test_single_linear_regression_r_squared(single_linear_regression_model):
    """
    Test regression model r_squared
    :return:
    """
    # Train Data
    train_r_squared = single_linear_regression_model.calculate_r_squared(
        single_linear_regression_model.independent_vars_train,
        single_linear_regression_model.dependent_var_train[:, 0])

    test_r_squared = single_linear_regression_model.calculate_r_squared(
        single_linear_regression_model.independent_vars_test,
        single_linear_regression_model.dependent_var_test[:, 0])

    assert(pytest.approx(train_r_squared, 0.001) == 0.9912525441776708)
    assert(pytest.approx(test_r_squared, 0.001) == 0.9988914111960758)
