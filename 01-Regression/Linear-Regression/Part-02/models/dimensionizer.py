import numpy as np
from models.data_handler import PreProcessData


class PrincipalComponentAnalysis(PreProcessData):
    def __init__(self, predictor_vars, response_var,
                 scale_type=None,
                 train_split=0.70, seed=123,
                 variance_explained_cutoff=0.95):
        """
        Returns object with PCA matrix and can be used to predict
        :param variance_explained:
        """
        self.variance_explained_cutoff = variance_explained_cutoff
        self.eigenvalues_all = []
        self.eigenvectors_all = []
        self.eigenvalues = []
        self.eigenvectors = []
        super().__init__(predictor_vars, response_var, scale_type, train_split, seed)
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
        self.eigenvectors = self.eigenvectors_all[:, :len(self.pct_var_exp_cumulative_all)]
        self.eigenvalues = self.eigenvalues_all[:len(self.pct_var_exp_cumulative_all)]

    def predict_pca(self, data: np.ndarray):
        return data.dot(self.eigenvectors)




