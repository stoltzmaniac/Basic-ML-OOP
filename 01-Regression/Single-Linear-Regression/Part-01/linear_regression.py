# All models for modeling


class SingleLinearRegression:

    def __init__(self, independent_var: list, dependent_var: list, n_predictions: int):

        """
        Completes either a single or multiple linear regression
        :param independent_var: list
        :param dependent_var: list
        """
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.n_predictions = n_predictions

    def fit(self) -> dict:
        pass

    def predict(self) -> dict:
        pass

    def __repr__(self):
        pass

    def __str__(self):
        return f"""
            This class returns a dictionary of results from your on your linear regression:
            {{
                'independent_var': {self.independent_var},
                'dependent_var': {self.dependent_var},
                'fit': {{
                    'coefficient': coefficent,
                    'constant': constant,
                    'r_squared': r_squared,
                    'p_values': 'p_values'
                    }},
                'n_predictions': {self.n_predictions}, 
                'predictions': {{
                    'x': [],
                    'y': []
                    }}
            }}
            :return: dict
            """
