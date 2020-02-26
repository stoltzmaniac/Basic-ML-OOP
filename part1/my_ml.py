# Classes to be used to host models and analyze data
from pprint import pprint
import pandas as pd
import altair as alt


class AnalyzeIris:
    def __init__(self, dataset: pd.DataFrame, model):
        self.dataset = dataset
        self.model = model

    def scatter_plot(self, x_axis, y_axis, colors):
        chart = alt.Chart(self.dataset).mark_point().encode(x=x_axis, y=y_axis, color=colors).interactive()
        chart.serve()

    def regression_model(self, dependent_var):
        numeric_data = self.dataset.select_dtypes(include=['float64', 'int'])
        indep_vars = list(set(numeric_data.columns) - {dependent_var})
        regr = self.model(endog=self.dataset[dependent_var], exog=self.dataset[indep_vars]).fit()
        print(regr.summary())

    def __str__(self):
        return "Hello, this is AnalyzeIris!"
