import numpy as np
import pytest

from models.regression import SingleLinearRegression


@pytest.fixture(scope='module')
def reg_model(single_linear_regression_data):
    linear_regression_model = SingleLinearRegression(independent_var=single_linear_regression_data['independent_var'],
                                                     dependent_var=single_linear_regression_data['response_var'])
    return linear_regression_model


def test_single_linear_regression_data_passing_correctly(reg_model, single_linear_regression_data):
    """
    Setup linear regression model
    :return:
    """
    assert(reg_model.independent_var.all() == single_linear_regression_data['independent_var'].all())
    assert(reg_model.dependent_var.all() == single_linear_regression_data['response_var'].all())
    assert(type(reg_model.independent_var) == np.ndarray)
    assert(type(reg_model.dependent_var) == np.ndarray)


def test_single_linear_regression_fit(reg_model):
    """
    Test regression model coefficients
    :return:
    """
    assert(pytest.approx(reg_model.b1, 0.01) == 1.14)
    assert(pytest.approx(reg_model.b0, 0.01) == 0.43)


def test_single_linear_regression_rmse(reg_model):
    """
    Test regression model root mean squared error
    :return:
    """
    assert(pytest.approx(reg_model.root_mean_squared_error(), 0.02) == 0.31)


def test_single_linear_regression_r_squared(reg_model):
    """
    Test regression model r_squared
    :return:
    """
    assert(pytest.approx(reg_model.r_squared(), 0.01) == 0.52)
