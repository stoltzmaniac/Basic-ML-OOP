import numpy as np
import pytest

from models.regression import LinearRegression


@pytest.fixture(scope="module")
def single_linear_regression_model(single_linear_regression_data):
    linear_regression = LinearRegression(
        predictor_vars=single_linear_regression_data["predictor_vars"],
        response_var=single_linear_regression_data["response_var"],
        train_split=0.7,
        seed=123,
        scale_type="normalize",
        learning_rate=0.01,
        tolerance=0.00001,
        batch_size=12,
        max_epochs=1000,
        decay=0.90,
    )

    linear_regression.fit_stochastic_gradient_descent()
    print(linear_regression)
    return linear_regression


def test_single_linear_regression_data_passing_correctly(
    single_linear_regression_model, single_linear_regression_data
):
    """
    Setup linear regression model
    :return:
    """
    assert (
        single_linear_regression_model.predictor_vars_train.all()
        == single_linear_regression_data["predictor_vars"].all()
    )
    assert (
        single_linear_regression_model.response_var_train.all()
        == single_linear_regression_data["response_var"].all()
    )
    assert type(single_linear_regression_model.predictor_vars_train) == np.ndarray
    assert type(single_linear_regression_model.response_var_train) == np.ndarray


def test_single_linear_regression_coefficients(single_linear_regression_model):
    """
    Test regression model coefficients
    :return:
    """
    print(single_linear_regression_model)
    expected_coefficients = [(0, 151.27), (1, 303.90)]
    no_of_betas = len(single_linear_regression_model.B)
    for n in range(no_of_betas):
        assert single_linear_regression_model.B[n] == pytest.approx(
            expected_coefficients[n][1], 0.001
        )


def test_single_linear_regression_r_squared(single_linear_regression_model):
    """
    Test regression model r_squared
    :return:
    """
    # Train Data
    train_r_squared = single_linear_regression_model.calculate_r_squared(
        single_linear_regression_model.predictor_vars_train,
        single_linear_regression_model.response_var_train[:, 0],
    )

    test_r_squared = single_linear_regression_model.calculate_r_squared(
        single_linear_regression_model.predictor_vars_test,
        single_linear_regression_model.response_var_test[:, 0],
    )

    assert pytest.approx(train_r_squared, 0.001) == 1
    assert pytest.approx(test_r_squared, 0.001) == 1
