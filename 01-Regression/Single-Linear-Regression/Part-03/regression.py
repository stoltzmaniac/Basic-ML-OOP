# All models for modeling
import numpy as np


class Regression:
    def __init__(self,
                 independent_vars: np.ndarray,
                 dependent_var: np.ndarray,
                 learning_rate: float = 0.01,
                 train_split: float = 0.7,
                 seed: int = 123):
        """
        :param independent_vars: np.ndarray
        :param dependent_var: np.array (one dimensional)
        :param learning_rate: float
        :param train_split: float (0 < value < 1)
        :param seed: int
        """
        # Check data types
        if type(seed) != int:
            raise ValueError(f'seed value not an int')
        type_check_arrays = [type(independent_vars) == np.ndarray, type(dependent_var) == np.ndarray]
        if not all(type_check_arrays):
            raise ValueError(f'Type(s) of data for Regression class are not accurate')
        if not type(learning_rate) == float and type(train_split) == float and train_split <= 1:
            raise ValueError(f'learning_rate or train_split not acceptable input(s)')
        # Check for single dimensional dependent_var
        if dependent_var.shape[0] != 1 or dependent_var.shape[1] != independent_vars.shape[0]:
            raise ValueError(f'Dimension(s) of data for Regression class are not accurate')

        all_data = np.concatenate((independent_vars, dependent_var.T), axis=1)
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

        self.predicted_values = None


class LinearRegression(Regression):

    def __init__(self, independent_vars, dependent_var, learning_rate, train_split, seed):
        """
        All inherited from Regression class
        """
        super().__init__(independent_vars, dependent_var, learning_rate, train_split, seed)

        self.b1 = None
        self.b0 = None
        self.fit()

    @property
    def independent_vars_train_mean(self):
        return np.mean(self.independent_vars_train)

    @property
    def dependent_var_train_mean(self):
        return np.mean(self.dependent_var_train)

    def fit(self):
        # Format: independent_vars_hat = b1*dependent_var + b0
        x_minus_mean = [x - self.independent_vars_train_mean for x in self.independent_vars_train]
        y_minus_mean = [y - self.dependent_var_train_mean for y in self.dependent_var_train]
        b1_numerator = sum([x * y for x, y in zip(x_minus_mean, y_minus_mean)])
        b1_denominator = sum([(x - self.independent_vars_train_mean) ** 2 for x in self.independent_vars_train])
        self.b1 = b1_numerator / b1_denominator
        self.b0 = self.dependent_var_train_mean - (self.b1 * self.independent_vars_train_mean)

    def predict(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict * self.b1 + self.b0
        return predicted_values

    def root_mean_squared_error(self):
        dependent_var_hat = self.predict(self.independent_vars_train)
        sum_of_res = np.sum((dependent_var_hat - self.dependent_var_train) ** 2)
        rmse = np.sqrt(sum_of_res / len(dependent_var_hat))
        return rmse

    def r_squared(self):
        dependent_var_hat = self.predict(self.independent_vars_train)
        sum_of_sq = np.sum((self.dependent_var_train - self.dependent_var_train_mean) ** 2)
        sum_of_res = np.sum((self.dependent_var_train - dependent_var_hat) ** 2)
        return 1 - (sum_of_res / sum_of_sq)

    def __str__(self):
        return f"""
            Model Results
            -------------
            b1: {np.round(self.b1, 2)}
            b0: {np.round(self.b0, 2)}
            RMSE: {np.round(self.root_mean_squared_error(), 2)}
            R^2: {np.round(self.r_squared(), 2)}
            """


# a = np.array([ [ 0,  1,  2],
#                [ 3,  4,  5],
#                [ 6,  7,  8],
#                [10, 11, 12],
#                [13, 14, 15],
#                [16, 17, 18]])
# b = np.array([['a','b','c','d','e','f']])
# a = np.array([ [ 0],
#                [ 3],
#                [ 6],
#                [10],
#                [13.9],
#                [16.1]])
# b = np.array([ [7,
#                5,
#                8.2,
#                12,
#                15.4,
#                8]])
#
# c = LinearRegression(independent_vars=a,
#                      dependent_var=b,
#                      learning_rate=1.0,
#                      train_split=0.67,
#                      seed=123)
#
# print(c)