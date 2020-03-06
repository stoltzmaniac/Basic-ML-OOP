# All models for modeling
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class Regression:
    def __init__(self, independent_vars: np.ndarray, dependent_var: np.ndarray):

        """
        Base class for regressions, can be utilized for single, multiple. Checks initial conditions of data.
        :param independent_vars: np.ndarray
        :param dependent_var: np.array (one dimensional)
        """
        # Check data types
        type_check = [type(independent_vars) == np.ndarray, type(dependent_var) == np.ndarray]
        if all(type_check):
            pass
        else:
            raise ValueError(f'Type(s) of data for Regression class are not accurate')
        # Check for single dimensional dependent_var
        if len(independent_vars.shape) < 1 or len(dependent_var.shape) != 1:
            raise ValueError(f'Dimension(s) of data for Regression class are not accurate')
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars


class SingleLinearRegression(Regression):
    """
    For single linear regression only, does not support multiple independent variables
    """
    def __init__(self, independent_vars, dependent_var):
        super().__init__(independent_vars, dependent_var)

        # independent_vars can only have 1 dimension
        if len(independent_vars.shape) != 1:
            raise ValueError(f'Dimension(s) of data for SingleLinearRegression class are not accurate')
        if len(independent_vars) != len(dependent_var):
            raise ValueError(f'Length of arrays do not match for SingleLinearRegression class are not accurate')
        self.independent_vars_mean = np.mean(self.independent_vars)
        self.dependent_var_mean = np.mean(self.dependent_var)
        self.b1 = None
        self.b0 = None
        self.predicted_values = None
        self.fit()

    def fit(self):
        # Format: y = b1*dependent_var + b0
        x_minus_mean = [x - self.independent_vars_mean for x in self.independent_vars]
        y_minus_mean = [y - self.dependent_var_mean for y in self.dependent_var]
        b1_numerator = sum([x * y for x, y in zip(x_minus_mean, y_minus_mean)])
        b1_denominator = sum([(x - self.independent_vars_mean) ** 2 for x in self.independent_vars])
        self.b1 = b1_numerator / b1_denominator
        self.b0 = self.dependent_var_mean - (self.b1 * self.independent_vars_mean)

    def predict(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict * self.b1 + self.b0
        return predicted_values

    def root_mean_squared_error(self):
        dependent_var_hat = self.predict(self.independent_vars)
        sum_of_res = np.sum((dependent_var_hat - self.dependent_var) ** 2)
        rmse = np.sqrt(sum_of_res / len(dependent_var_hat))
        return rmse

    def r_squared(self):
        dependent_var_hat = self.predict(self.independent_vars)
        sum_of_sq = np.sum((self.dependent_var - self.dependent_var_mean) ** 2)
        sum_of_res = np.sum((self.dependent_var - dependent_var_hat) ** 2)
        return 1 - (sum_of_res / sum_of_sq)

    def plot(self, min_independent_vars=None, max_independent_vars=None, increment=100,
             title=None, x_label=None, y_label=None, point_color='blue', line_color='red'):
        # If plot min / max not given, define based off of min and max of data
        if not min_independent_vars:
            min_independent_vars = np.min(self.independent_vars)
        if not max_independent_vars:
            max_independent_vars = np.max(self.independent_vars)
        # Set x-axis range
        independent_vars_hat = np.linspace(min_independent_vars, max_independent_vars, increment)
        dependent_var_hat = self.predict(independent_vars_hat)
        plt.plot(independent_vars_hat, dependent_var_hat, color=line_color)
        plt.scatter(self.independent_vars, self.dependent_var, color=point_color)
        plt.text(x=min(self.independent_vars)*1.1, y=max(self.dependent_var)*0.9,
                 bbox=dict(),
                 s=f'b1: {round(self.b1, 2)}\n'
                   f'b0: {round(self.b0, 2)}\n'
                   f'RMSE: {round(self.root_mean_squared_error(), 2)}\n'
                   f'R^2: {round(self.r_squared(), 2)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def __str__(self):
        return f"""
            Model Results
            -------------
            b1: {round(self.b1, 2)}\n
            b0: {round(self.b0, 2)}\n
            RMSE: {round(self.root_mean_squared_error(), 2)}\n
            R^2: {round(self.r_squared(), 2)}
            """
