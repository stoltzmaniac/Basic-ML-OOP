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

        super().__init__()

        self.predictor_vars = self.convert_dataframe_to_array(predictor_vars)
        self.response_var = self.convert_dataframe_to_array(response_var)

        if self.response_var.shape[0] != self.predictor_vars.shape[0]:
            raise ValueError(
                f"Dimension(s) of InputData class are not accurate\n"
                f"predictor_vars shape: {self.predictor_vars.shape}\n"
                f"response_var shape: {self.response_var.shape}"
            )

    @staticmethod
    def convert_dataframe_to_array(df_or_series):
        """
        Converts pandas dataframe to array if necessary
        :param df_or_series:
        :return: np.array
        """
        if isinstance(df_or_series, pd.DataFrame) or isinstance(df_or_series, pd.Series):
            dat = np.array(df_or_series)
            if len(dat.shape) == 1:
                return dat[:, np.newaxis]
            else:
                return dat
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
    def __init__(self, predictor_vars, response_var, train_split: float, seed: int = 5):
        """
        Split the input data to test / train split to be used in machine learning
        :param train_split: float percent used as training (Between 0 and 1)
        :param seed: int for repeatability
        """

        super().__init__(predictor_vars, response_var)

        self.seed = seed
        self.train_split = train_split

        if type(self.seed) != int:
            raise ValueError(f"seed value not an int, it is {self.seed}")
        if not type(self.train_split) == float:
            raise ValueError(f"train_split not a float, it is {self.train_split}")
        if not 0 < self.train_split < 1:
            raise ValueError(f"train_split must be between 0 and 1 (not including), it is {self.train_split}")

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
                 train_split, scale_type: str, seed):
        # TODO: add something to handle for categorical variables
        """
        Scales the data
        :param scale_type: str -> 'normalize', 'standardize', 'min_max', 'scale'
        """

        super().__init__(predictor_vars, response_var, train_split, seed)

        self.scale_type = scale_type

        if self.scale_type not in ['normalize', 'standardize', 'min_max', 'scale']:
            if self.scale_type:
                raise ValueError(f"scale_type {self.scale_type} not in ['normalize', 'standardize', 'min_max', 'scale']")
            else:
                pass

        self.predictor_mean = np.mean(self.predictor_vars_train, axis=0)
        self.predictor_std = np.std(self.predictor_vars_train, axis=0)
        self.predictor_max = np.max(self.predictor_vars_train, axis=0)
        self.predictor_min = np.min(self.predictor_vars_train, axis=0)

        self.predictor_vars_train = self.scale(data=self.predictor_vars_train)
        self.predictor_vars_test = self.scale(data=self.predictor_vars_test)

    def scale(self, data: np.ndarray):
        """
        Preprocess utilizing training data only. Will need this step for any additional modeling
        :param data: np.ndarray
        :param scale_type: str -> 'normalize', 'standardize', 'min_max', 'scale'
        :return:
        """
        if self.scale_type == 'min_max':
            scaled_data = (data - self.predictor_min) / (self.predictor_max - self.predictor_mean)
        elif self.scale_type == 'normalize':
            scaled_data = (data - self.predictor_mean) / (self.predictor_max - self.predictor_min)
        elif self.scale_type == 'standardize':
            scaled_data = (data - self.predictor_mean) / self.predictor_std
        elif self.scale_type == 'scale':
            scaled_data = data - self.predictor_mean
        else:
            scaled_data = data
        return scaled_data
