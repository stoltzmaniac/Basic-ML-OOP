# All models for modeling
import numpy as np
import matplotlib.pyplot as plt


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


# x = np.array([[1,3,4], [8,9,10]])
# y = np.array([4,5,6])
# a = Regression(x, y)


class SingleLinearRegression(Regression):
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
        plt.text(x=min(self.independent_vars), y=max(self.dependent_var),
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

#
# x = np.array([1,2,3])
# y = np.array([4,5,6])
# b = SingleLinearRegression(x, y)
# b.plot()
# print(b.b1)
# print(b.b0)
# print(b.root_mean_squared_error())
# print(b.r_squared())
#
# x = np.array([1,2,3,4,3,7,12,5,23,10])
# y = np.array([4,5,6,2,3,1,12,4,11,15])
# b = SingleLinearRegression(x, y)
# b.plot(title='Hello World', x_label='My X-Axis', y_label='My Y-Axis',
#        point_color='green', line_color='black')
# print(b.b1)
# print(b.b0)
# print(b.root_mean_squared_error())
# print(b.r_squared())
# pandas --> np.array(df[['independent_var', 'dependent_var']])
#
#
#     self.results = {
#         'independent_var': self.independent_var,
#         'dependent_var': self.dependent_var,
#         'fit': {
#             'coefficient': None,
#             'constant': None,
#             'r_squared': None,
#             'p_values': None
#         },
#         'predictions': {
#             'predict': self.predict,
#             'result': None
#         }
#     }
#
# def fit(self) -> dict:
#     """
#     Fit the linear regression to format y = b1*x + b0
#     :return:
#     """
#     coefficients = self.calculate_coefficient_and_constant()
#     self.results['fit']['coefficient'] = coefficients['b1']
#     self.results['fit']['constant'] = coefficients['b0']
#
# def predictions(self) -> dict:
#     pass
#
# def calculate_coefficient_and_constant(self) -> dict:
#     x_mean = mean(self.independent_var)
#     y_mean = mean(self.dependent_var)
#     x_minus_mean = [x - x_mean for x in self.independent_var]
#     y_minus_mean = [y - y_mean for y in self.dependent_var]
#     b1_numerator = sum([x * y for x, y in zip(x_minus_mean, y_minus_mean)])
#     b1_denominator = sum([(x - x_mean) ** 2 for x in self.independent_var])
#     b1 = b1_numerator / b1_denominator
#     b0 = y_mean - (b1 * x_mean)
#     return {'b1': b1, 'b0': b0}
#
# def calculate_r_squared_and_p_values(self) -> dict:
#     y_fit_values = [x * self.results['fit']['coefficient'] + self.results['fit']['constant'] for x in self.independent_var]
#     print(y_fit_values)
#
#
# def __str__(self):
#     return f"""
#         This class returns a dictionary of results from your on your linear regression:
#         {{
#             'independent_var': {self.independent_var},
#             'dependent_var': {self.dependent_var},
#             'fit': {{
#                 'coefficient': coefficient,
#                 'constant': constant,
#                 'r_squared': r_squared,
#                 'p_values': 'p_values'
#                 }},
#             'predictions': {{
#                 'predict': {self.predict},
#                 'result': result_of_predictions.
#                 }}
#         }}
#         :return: dict
#         """
