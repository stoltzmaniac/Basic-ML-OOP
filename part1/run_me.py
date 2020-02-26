import click

from models import sm
from datasets import iris
from my_ml import AnalyzeIris


@click.command()
@click.option('-p', '--print_results', default=1, help='Prints regression results, 0 or 1.')
@click.option('-s', '--scatter', default=0, help='Creates a scatter plot, 0 or 1.')
@click.option('-x', '--x_axis', default='petalLength', help='IF plotting, column name to be used as x-axis.')
@click.option('-y', '--y_axis', default='petalWidth', help='IF plotting, column name to be used as y-axis.')
@click.option('-c', '--colors', default='species', help='IF plotting, column name to breakdown color of points.')
def main(x_axis, y_axis, colors, scatter, print_results):
    print('Starting my_ml.py')

    # Create an instance with data and linear model
    model = AnalyzeIris(iris, sm.OLS)

    if scatter:
        model.scatter_plot(x_axis=x_axis, y_axis=y_axis, colors=colors)

    if print_results:
        model.regression_model(dependent_var=y_axis)

    print('Finished running my_ml.py')


if __name__ == '__main__':
    main()
