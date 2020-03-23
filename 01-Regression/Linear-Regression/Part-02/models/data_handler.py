import numpy as np
import pandas as pd


class InputBase:
    def __init__(self):
        """
        Base class
        """
        pass


class InputData(InputBase):
    def __init__(self, predictor_vars: np.ndarray, response_var: np.ndarray):
        """
        Houses the raw data. Data should be input as a numpy array, but if it
        is in a pandas.Dataframe or pandas.Series it will be converted.
        :param predictor_vars: np.ndarray
        :param response_var: np.array (one dimensional)
        """
        self.predictor_vars = self.convert_dataframe_to_array(predictor_vars)
        self.response_var = self.convert_dataframe_to_array(response_var)

        if self.response_var.shape[0] != self.predictor_vars.shape[0]:
            raise ValueError(
                f"Dimension(s) of InputData class are not accurate\n"
                f"predictor_vars shape: {self.predictor_vars.shape}\n"
                f"response_var shape: {self.response_var.shape}"
            )
        super().__init__()

    @staticmethod
    def convert_dataframe_to_array(df_or_series):
        """
        Converts pandas dataframe to array if necessary
        :param df_or_series:
        :return: np.array
        """
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


class SplitTestTrain(InputData):
    def __init__(self, predictor_vars, response_var, train_split=0.70, seed=123):
        """
        Split the input data to test / train split to be used in machine learning
        :param predictor_vars: np.ndarray
        :param response_var: np.ndarray
        :param train_split: float percent used as training (Between 0 and 1)
        :param seed: int for repeatability
        """
        self.seed = seed
        self.train_split = train_split
        super().__init__(predictor_vars, response_var)
        self.split_data()

    def split_data(self):
        """
        Separates the data and splits it accordingly, utilizes the random seed
        :return:
        """
        np.random.seed(seed=self.seed)
        indices = np.random.permutation(self.predictor_vars.shape[0])
        split_row = round(self.predictor_vars.shape[0] * self.train_split)
        train_idx, test_idx = indices[:split_row], indices[split_row:]
        self.predictor_vars_train, self.predictor_vars_test = self.predictor_vars[train_idx, :], self.predictor_vars[test_idx, :]
        self.response_var_train, self.response_var_test = self.response_var[train_idx], self.response_var[test_idx]


class PreProcessData(SplitTestTrain):
    def __init__(self, predictor_vars, response_var,
                 train_split=0.70, seed=123,
                 scale=None):
        """
        Split the input data to test / train split to be used in machine learning
        :param predictor_vars: np.ndarray
        :param response_var: np.ndarray
        :param train_split: float percent used as training (Between 0 and 1)
        :param seed: int for repeatability
        :param scale: str -> 'normalize', 'standardize', 'min_max', 'scale'
        """
        self.scale = scale
        self.predictor_vars_train_scale = None
        super().__init__(predictor_vars, response_var, train_split, seed)
        self.preprocess()

    def preprocess(self):
        predictor_mean = np.mean(self.predictor_vars_train, axis=0)
        predictor_std = np.std(self.predictor_vars_train, axis=0)
        predictor_max = np.max(self.predictor_vars_train, axis=0)
        predictor_min = np.min(self.predictor_vars_train, axis=0)
        if self.scale == 'min_max':
            self.predictor_vars_train_scale = (self.predictor_vars_train - predictor_min) / (predictor_max - predictor_mean)
        elif self.scale == 'normalize':
            self.predictor_vars_train_scale = (self.predictor_vars_train - predictor_mean) / (predictor_max-predictor_min)
        elif self.scale == 'standardize':
            self.predictor_vars_train_scale = (self.predictor_vars_train - predictor_mean) / predictor_std
        elif self.scale == 'scale':
            self.predictor_vars_train_scale = self.predictor_vars_train - predictor_mean
        else:
            self.predictor_vars_train_scale = self.predictor_vars_train


# #normalize
# x = np.array(data)
# x_mean = np.mean(x, axis=0)
# x_std = np.std(x, axis=0)
#
# x1 = x - x_mean
# x2 = x1 / x_std