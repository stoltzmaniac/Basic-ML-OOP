# Code to execute from cli
import csv
import click

import numpy as np

from linear_regression import SingleLinearRegression


@click.command()
@click.option('-d', '--dataset', default='./data/fake_data.csv',
              help='Dataset with independent variable in first column and dependent variable in second. \
              Dataset has a header row.')
def main(dataset: str):
    print('Starting run_me.py')

    # Read in csv data
    independent_data = []
    dependent_data = []
    with open(dataset, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Removes header row
        for row in reader:
            independent_data.append(float(row[0]))
            dependent_data.append(float(row[1]))
    x = np.array(independent_data)
    y = np.array(dependent_data)
    # Create instance of SingleLinearRegression model
    single_linear_regression = SingleLinearRegression(
        independent_vars=x,
        dependent_var=y)

    print(single_linear_regression)

    single_linear_regression.plot(title='Hello World', x_label='My X-Axis', y_label='My Y-Axis',
           point_color='green', line_color='black')


if __name__ == '__main__':
    main()