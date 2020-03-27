import numpy as np
import pytest

from models.dimensionizers import PrincipalComponentAnalysis


@pytest.fixture(scope="module")
def principal_component_analysis(iris_data):

    pca = PrincipalComponentAnalysis(
        predictor_vars=iris_data['predictor_vars'],
        response_var=iris_data['response_var'],
        scale_type='normalize',
        train_split=0.7,
        seed=123,
        variance_explained_cutoff=0.95
    )

    print(pca)
    return pca


def test_principal_component_analysis_results(
    principal_component_analysis
):
    """
    Test PCA results
    :return:
    """
    assert principal_component_analysis.variance_explained_cutoff == 0.95
    assert pytest.approx(principal_component_analysis.pct_var_exp_cumulative, 0.01) == [0.8414]
    assert pytest.approx(principal_component_analysis.eigenvalues, 0.01) == [0.24742686]
    assert all(principal_component_analysis.eigenvectors) == all([[0.43061003], [-0.15830383], [0.61817815], [0.63825597]])
