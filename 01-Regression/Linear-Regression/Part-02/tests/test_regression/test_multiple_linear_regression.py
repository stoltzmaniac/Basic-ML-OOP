import numpy as np
import pytest

from models.regression import LinearRegression


@pytest.fixture(scope="module")
def multiple_linear_regression_model(multiple_linear_regression_data):
    linear_regression = LinearRegression(predictor_vars=multiple_linear_regression_data["predictor_vars"],
                                         response_var=multiple_linear_regression_data["response_var"],
                                         train_split=0.7,
                                         seed=123,
                                         learning_rate=0.01,
                                         scale_type='normalize',
                                         tolerance=0.00001,
                                         batch_size=12,
                                         max_epochs=1000,
                                         decay=0.90)

    linear_regression.fit_stochastic_gradient_descent()
    print(linear_regression)
    return linear_regression


def test_multiple_linear_regression_data_passing_correctly(
        multiple_linear_regression_model, multiple_linear_regression_data
):
    """
    Setup linear regression model
    :return:
    """

    assert (
            multiple_linear_regression_model.predictor_vars_train.all()
            == np.array(multiple_linear_regression_data["predictor_vars"]).all()
    )
    assert (
            multiple_linear_regression_model.response_var_train.all()
            == multiple_linear_regression_data["response_var"].all()
    )
    assert type(multiple_linear_regression_model.predictor_vars_train) == np.ndarray
    assert type(multiple_linear_regression_model.response_var_train) == np.ndarray


def test_multiple_linear_regression_coefficients(multiple_linear_regression_model):
    """
    Test regression model coefficients
    :return:
    """
    print(multiple_linear_regression_model)
    expected_coefficients = [
        (0, 0.999),
        (1, 0.111),
        (2, 0.358),
        (3, 1.000),
    ]
    no_of_betas = len(multiple_linear_regression_model.B)
    for n in range(no_of_betas):
        assert (
                pytest.approx(expected_coefficients[n][1], 0.01)
                == multiple_linear_regression_model.B[n]
        )


def test_multiple_linear_regression_r_squared(multiple_linear_regression_model):
    """
    Test regression model r_squared
    :return:
    """
    # Train Data
    train_r_squared = multiple_linear_regression_model.calculate_r_squared(
        multiple_linear_regression_model.predictor_vars_train,
        multiple_linear_regression_model.response_var_train[:, 0],
    )

    test_r_squared = multiple_linear_regression_model.calculate_r_squared(
        multiple_linear_regression_model.predictor_vars_test,
        multiple_linear_regression_model.response_var_test[:, 0],
    )

    assert pytest.approx(train_r_squared, 0.001) == 0.62301
    assert pytest.approx(test_r_squared, 0.001) == 0.42328
