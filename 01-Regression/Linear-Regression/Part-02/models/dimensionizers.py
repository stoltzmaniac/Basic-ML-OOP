import numpy as np
from models.data_handler import PreProcessData


class PrincipalComponentAnalysis(PreProcessData):
    def __init__(self, predictor_vars, response_var,
                 train_split,
                 seed,
                 scale_type,
                 variance_explained_cutoff: float):
        """
        Returns object with PCA matrix and can be used to predict
        :param variance_explained_cutoff: float with value between 0 and 1, max cumulative variance explained cutoff
        """
        self.variance_explained_cutoff = variance_explained_cutoff
        self.eigenvalues_all = []
        self.eigenvectors_all = []
        self.eigenvalues = []
        self.eigenvectors = []
        self.pca_predictor_vars = np.ndarray

        if type(self.variance_explained_cutoff) != float or not 0 < self.variance_explained_cutoff < 1:
            raise ValueError(f"variance_explained_cutoff needs to be a float between 0 and 1, it is {self.variance_explained_cutoff}")

        super().__init__(predictor_vars, response_var, train_split, seed, scale_type)

        self.calculate_eigens()

    def calculate_eigens(self):
        """
        Create eigenvalues and eigenvectors in descending order, check cutoff
        :return:
        """
        covariance_matrix = np.cov(self.predictor_vars_train.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        # Create "All" version
        self.eigenvalues_all = eigenvalues[idx]
        self.eigenvectors_all = eigenvectors[:, idx]
        # Create selected percentage version with cutoff
        eigenvalues_pct = self.eigenvalues_all / np.sum(self.eigenvalues_all)
        self.pct_var_exp_cumulative_all = np.cumsum(eigenvalues_pct)
        self.pct_var_exp_cumulative = self.pct_var_exp_cumulative_all[self.pct_var_exp_cumulative_all <= self.variance_explained_cutoff]
        self.eigenvectors = self.eigenvectors_all[:, :len(self.pct_var_exp_cumulative)]
        self.eigenvalues = self.eigenvalues_all[:len(self.pct_var_exp_cumulative)]

    def build(self, data: np.ndarray):
        """
        Converts outside of the train set
        :param data: np.ndarray
        :return:
        """
        ret = data.dot(self.eigenvectors)
        self.pca_predictor_vars = ret
        return ret

    def __str__(self):
        return f"""
        Variance Explained Cutoff: {self.variance_explained_cutoff}
        PCA Variance Explained: {self.pct_var_exp_cumulative}
        Eigenvalues: 
        {self.eigenvalues}
        Eigenvectors: 
        {self.eigenvectors}
        """