# All models for modeling
import numpy as np

class SingleLinearRegression:

    def __init__(self, independent_var: np.array, dependent_var: np.array):
        """
        Complete a single linear regression.
        :param independent_var: list
        :param dependent_var: list
        """
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.b1 = None
        self.b0 = None
        self.predicted_values = None
        self.fit()

    @property
    def independent_var_mean(self):
        return np.mean(self.independent_var)

    @property
    def dependent_var_mean(self):
        return np.mean(self.dependent_var)

    def fit(self):
        # Format: independent_var_hat = b1*dependent_var + b0
        x_minus_mean = [x - self.independent_var_mean for x in self.independent_var]
        y_minus_mean = [y - self.dependent_var_mean for y in self.dependent_var]
        b1_numerator = sum([x * y for x, y in zip(x_minus_mean, y_minus_mean)])
        b1_denominator = sum([(x - self.independent_var_mean) ** 2 for x in self.independent_var])
        self.b1 = b1_numerator / b1_denominator
        self.b0 = self.dependent_var_mean - (self.b1 * self.independent_var_mean)

    def predict(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict * self.b1 + self.b0
        return predicted_values

    def root_mean_squared_error(self):
        dependent_var_hat = self.predict(self.independent_var)
        sum_of_res = np.sum((dependent_var_hat - self.dependent_var) ** 2)
        rmse = np.sqrt(sum_of_res / len(dependent_var_hat))
        return rmse

    def r_squared(self):
        dependent_var_hat = self.predict(self.independent_var)
        sum_of_sq = np.sum((self.dependent_var - self.dependent_var_mean) ** 2)
        sum_of_res = np.sum((self.dependent_var - dependent_var_hat) ** 2)
        return 1 - (sum_of_res / sum_of_sq)

    def __str__(self):
        return f"""
            Model Results
            -------------
            b1: {round(self.b1, 2)}
            b0: {round(self.b0, 2)}
            RMSE: {round(self.root_mean_squared_error(), 2)}
            R^2: {round(self.r_squared(), 2)}
            """



