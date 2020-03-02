# All models for modeling


class SingleLinearRegression:

    def __init__(self, dependent_var: list, independent_var: list, n_predictions: int):

        """
        Completes either a single or multiple linear regression
        :param dependent_var: list
        :param independent_var: list
        """
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.n_predictions = n_predictions
        pass

    def fit(self) -> dict:
        pass

    def predict(self) -> dict:
        if self.n_predictions:
            pass
        else:
            pass

    def __repr__(self):
        pass

    def __str__(self):
        """
        This class returns a dictionary of results from your on your linear regression:
        {
            'dependent_var': [dependent_var],
            'independent_var': [independent_var],
            'fit': {
                'coefficient': coefficent,
                'constant': constant,
                'r_squared': r_squared,
                'p_values': 'p_values'
                },
            predictions: {
                'x': [],
                'y': []
                }
        }
        :return: dict
        """


class MachineLearning(SingleLinearRegression):
    pass
