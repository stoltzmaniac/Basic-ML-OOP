import numpy as np
import pytest

from models.regression import LinearRegression


@pytest.fixture(scope="module")
def multiple_linear_regression_model(multiple_linear_regression_data):
    linear_regression_model = LinearRegression(
        independent_vars=multiple_linear_regression_data["independent_vars"],
        dependent_var=multiple_linear_regression_data["dependent_var"],
        iterations=10000,
        learning_rate=0.001,
        train_split=0.7,
        seed=123,
    )
    return linear_regression_model


def test_multiple_linear_regression_data_passing_correctly(
    multiple_linear_regression_model, multiple_linear_regression_data
):
    """
    Setup linear regression model
    :return:
    """
    assert (
        multiple_linear_regression_model.independent_vars_train.all()
        == multiple_linear_regression_data["independent_vars"].all()
    )
    assert (
        multiple_linear_regression_model.dependent_var_train.all()
        == multiple_linear_regression_data["dependent_var"].all()
    )
    assert type(multiple_linear_regression_model.independent_vars_train) == np.ndarray
    assert type(multiple_linear_regression_model.dependent_var_train) == np.ndarray


def test_multiple_linear_regression_coefficients(multiple_linear_regression_model):
    """
    Test regression model coefficients
    :return:
    """
    print(multiple_linear_regression_model)
    expected_coefficients = [
        (0, 0.31701131823834194),
        (1, 0.019291066504884043),
        (2, 0.03215158052302674),
        (3, 0.7237572134362669),
    ]
    no_of_betas = len(multiple_linear_regression_model.B)
    for n in range(no_of_betas):
        assert (
            pytest.approx(expected_coefficients[n][1], 0.001)
            == multiple_linear_regression_model.B[n]
        )


def test_multiple_linear_regression_r_squared(multiple_linear_regression_model):
    """
    Test regression model r_squared
    :return:
    """
    # Train Data
    train_r_squared = multiple_linear_regression_model.calculate_r_squared(
        multiple_linear_regression_model.independent_vars_train,
        multiple_linear_regression_model.dependent_var_train[:, 0],
    )

    test_r_squared = multiple_linear_regression_model.calculate_r_squared(
        multiple_linear_regression_model.independent_vars_test,
        multiple_linear_regression_model.dependent_var_test[:, 0],
    )

    assert pytest.approx(train_r_squared, 0.001) == 0.5179372162803452
    assert pytest.approx(test_r_squared, 0.001) == 0.2796260458401886
