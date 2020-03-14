# All models for modeling
import numpy as np


class Regression:
    def __init__(self,
                 independent_vars: np.ndarray,
                 dependent_var: np.ndarray,
                 iterations: int = 1000,
                 learning_rate: float = 0.01,
                 train_split: float = 0.7,
                 seed: int = 123):
        """
        :param independent_vars: np.ndarray
        :param dependent_var: np.array (one dimensional)
        :param iterations: int
        :param learning_rate: float
        :param train_split: float (0 < value < 1)
        :param seed: int
        """
        # Check data types
        if type(seed) != int:
            raise ValueError(f'seed value not an int')
        if type(iterations) != int or iterations <= 0:
            raise ValueError(f'Invalid iterations value')
        type_check_arrays = [type(independent_vars) == np.ndarray, type(dependent_var) == np.ndarray]
        if not all(type_check_arrays):
            raise ValueError(f'Type(s) of data for Regression class are not accurate')
        if not type(learning_rate) == float and type(train_split) == float and train_split <= 1:
            raise ValueError(f'learning_rate or train_split not acceptable input(s)')
        if dependent_var.shape[0] != independent_vars.shape[0]:
            raise ValueError(f'Dimension(s) of data for Regression class are not accurate')

        all_data = np.column_stack((independent_vars, dependent_var))
        # all_data = np.concatenate((independent_vars, dependent_var.T), axis=1)
        np.random.seed(seed)
        np.random.shuffle(all_data)

        split_row = round(all_data.shape[0] * train_split)
        train_data = all_data[:split_row]
        test_data = all_data[split_row:]

        # Train
        self.independent_vars_train = train_data[:, :-1]
        self.dependent_var_train = train_data[:, -1:]

        # Test
        self.independent_vars_test = test_data[:, :-1]
        self.dependent_var_test = test_data[:, -1:]

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost = []


class LinearRegression(Regression):

    def __init__(self, independent_vars, dependent_var, iterations, learning_rate, train_split, seed):
        """
        All inherited from Regression class
        """
        super().__init__(independent_vars, dependent_var, iterations, learning_rate, train_split, seed)

        # Initialize betas
        if len(self.independent_vars_train.shape) == 1:
            self.B = np.zeros(1)
        else:
            self.B = np.zeros(self.independent_vars_train.shape[1])

        # Automatically fit
        self.fit_gradient_descent()

    def fit_gradient_descent(self):
        len_dep_var_train = len(self.dependent_var_train[:, 0])
        for i in range(self.iterations):
            loss = self.independent_vars_train.dot(self.B) - self.dependent_var_train[:, 0]
            gradient = self.independent_vars_train.T.dot(loss) / len_dep_var_train
            self.B = self.B - (gradient * self.learning_rate)
            cost = np.sum((self.independent_vars_train.dot(self.B) - self.dependent_var_train[:, 0]) ** 2) / (2 * len_dep_var_train)
            self.cost.append(cost)

    def predict(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict.dot(self.B)
        return predicted_values

    @property
    def r_squared(self):
        sum_sq_r = np.sum((self.predict(self.independent_vars_train) - self.dependent_var_train[:, 0]) ** 2)
        sum_sq_t = np.sum((self.dependent_var_train[:, 0] - self.dependent_var_train[:, 0].mean()) ** 2)
        return 1 - (sum_sq_r / sum_sq_t)

    def __str__(self):
        return f"""
            Model Results
            -------------
            Betas: {self.B}
            R^2: {self.r_squared}
            """

# Example
# import pandas as pd
# import numpy as np
# from regression import LinearRegression, Regression
#
# # Single
# data = pd.read_csv('/Users/stoltzmanconsulting/Documents/Git-Repositories/GitHub/Basic-ML-OOP/01-Regression/Linear-Regression/Part-01/tests/my_test_data/my_test_data_2.csv')
# single_linear_regression = LinearRegression(
#     independent_vars=np.array(data)[:, :1],
#     dependent_var=np.array(data)[:, -1],
#     iterations=1000,
#     learning_rate=0.001,
#     train_split=0.7,
#     seed=123
# )
#
# print(single_linear_regression)
#
#
# # Multiple
# data = pd.read_csv('/Users/stoltzmanconsulting/Documents/Git-Repositories/GitHub/Basic-ML-OOP/01-Regression/Linear-Regression/Part-01/tests/my_test_data/my_test_data.csv')
# multiple_linear_regression = LinearRegression(
#     independent_vars=np.array(data)[:, :3],
#     dependent_var=np.array(data)[:, -1],
#     iterations=1000,
#     learning_rate=0.001,
#     train_split=0.7,
#     seed=123
# )
#
# print(multiple_linear_regression)
