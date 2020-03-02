# Code to execute from cli
import csv
import click
from linear_regression import SingleLinearRegression


@click.command()
@click.option('-d', '--dataset', default='./data/fake_data.csv',
              help='Dataset with independent variable in first column and dependent variable in second. \
              Dataset has a header row.')
@click.option('-p', '--predict', default=2.5,
              help='Dependent variable value you would like to use the fit to predict.')
def main(dataset: str, predict: int):
    print('Starting run_me.py')

    # Read in csv data
    independent_data = []
    dependent_data = []
    with open(dataset, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Removes header row
        for row in reader:
            independent_data.append(row[0])
            dependent_data.append(row[1])

    # Create instance of SingleLinearRegression model
    single_linear_regression = SingleLinearRegression(
        independent_var=independent_data,
        dependent_var=dependent_data,
        predict=predict
    )

    print(single_linear_regression)


if __name__ == '__main__':
    main()
    print('Finished run_me.py')
    input("Press enter to exit")
