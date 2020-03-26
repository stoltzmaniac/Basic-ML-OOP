# All models for modeling


class SingleLinearRegression:

    def __init__(self, independent_var: list, dependent_var: list, predict: float):

        """
        Completes either a single or multiple linear regression. We will pass a single value to predict.
        :param independent_var: list
        :param dependent_var: list
        :param predict: float
        """
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.predict = predict

    def fit(self) -> dict:
        pass

    def predictions(self) -> dict:
        pass

    def __str__(self):
        return f"""
            This class returns a dictionary of results from your on your linear regression:
            {{
                'independent_var': {self.independent_var},
                'response_var': {self.dependent_var},
                'fit': {{
                    'coefficient': coefficient,
                    'constant': constant,
                    'r_squared': r_squared,
                    'p_values': 'p_values'
                    }}, 
                'predictions': {{
                    'predict': {self.predict},
                    'result': result_of_predictions.
                    }}
            }}
            :return: dict
            """
