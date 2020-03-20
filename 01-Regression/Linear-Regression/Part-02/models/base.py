import numpy as np
import pandas as pd


class InputBase:
    def __init__(self):
        pass


class RawInputData(InputBase):
    def __init__(self, predictor_vars: np.ndarray, response_var: np.ndarray):
        """
        :param predictor_vars: np.ndarray
        :param response_var: np.array (one dimensional)
        """
        self.predictor_vars = self.convert_to_array(predictor_vars)
        self.response_var = self.convert_to_array(response_var)

        if self.response_var.shape[0] != self.predictor_vars.shape[0]:
            raise ValueError(
                f"Dimension(s) of InputData class are not accurate\n"
                f"predictor_vars shape: {self.predictor_vars.shape}\n"
                f"response_var shape: {self.response_var.shape}"
            )
        super().__init__()

    @staticmethod
    def convert_to_array(df_or_series):
        if isinstance(df_or_series, pd.DataFrame) or isinstance(df_or_series, pd.Series):
            return np.array(df_or_series)
        if isinstance(df_or_series, np.ndarray):
            return df_or_series
        else:
            raise TypeError(f"InputData error:\n"
                            f"type should be of np.ndarray and is currently type: {type(df_or_series)}")

    def __str__(self):
        return f'-----First 10 Rows----\n' \
               f'Independent Variables:\n' \
               f'{self.predictor_vars[0:10,]}\n' \
               f'-----------------------\n' \
               f'Dependent Variable:\n' \
               f'{self.response_var[0:10,]}\n' \
               f'-----------------------\n'


class SplitTestTrain(RawInputData):
    def __init__(self, predictor_vars, response_var, train_split=0.70, seed=123):
        self.seed = seed
        self.train_split = train_split
        super().__init__(predictor_vars, response_var)
        self.split_data()

    def split_data(self):
        np.random.seed(seed=self.seed)
        indices = np.random.permutation(self.predictor_vars.shape[0])
        split_row = round(self.predictor_vars.shape[0] * self.train_split)
        train_idx, test_idx = indices[:split_row], indices[split_row:]
        self.predictor_vars_train, self.predictor_vars_test = self.predictor_vars[train_idx, :], self.predictor_vars[test_idx, :]
        self.response_var_train, self.response_var_test = self.response_var[train_idx], self.response_var[test_idx]
