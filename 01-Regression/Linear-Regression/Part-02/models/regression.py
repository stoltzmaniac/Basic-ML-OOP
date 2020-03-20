# All models for modeling
import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(
        self,
        independent_vars: np.ndarray,
        dependent_var: np.ndarray,
        iterations: int,
        learning_rate: float,
        train_split: float,
        seed: int,
        plot_style: str
    ):
        """
        :param independent_vars: np.ndarray
        :param dependent_var: np.array (one dimensional)
        :param iterations: int
        :param learning_rate: float
        :param train_split: float (0 < value < 1)
        :param seed: int
        :param plot_style: str (ex. 'fivethirtyeight')
        """
        # Check data types
        if type(seed) != int:
            raise ValueError(f"seed value not an int")
        if type(plot_style) != str:
            raise ValueError(f"plot_style not a str")
        if type(iterations) != int or iterations <= 0:
            raise ValueError(f"Invalid iterations value")
        type_check_arrays = [
            type(independent_vars) == np.ndarray,
            type(dependent_var) == np.ndarray,
        ]
        if not all(type_check_arrays):
            raise ValueError(f"Type(s) of data for Regression class are not accurate")
        if (
            not type(learning_rate) == float
            and type(train_split) == float
            and train_split <= 1
        ):
            raise ValueError(f"learning_rate or train_split not acceptable input(s)")
        if dependent_var.shape[0] != independent_vars.shape[0]:
            raise ValueError(
                f"Dimension(s) of data for Regression class are not accurate"
            )

        plt.style.use(plot_style)

        all_data = np.column_stack((independent_vars, dependent_var))
        all_data = all_data.astype("float")
        np.random.seed(seed)
        np.random.shuffle(all_data)

        split_row = round(all_data.shape[0] * train_split)
        train_data = all_data[:split_row]
        test_data = all_data[split_row:]

        # Train
        self.independent_vars_train = train_data[:, :-1]
        self.dependent_var_train = train_data[:, -1:]

        # Test
        self.independent_vars_test = test_data[:, :-1]
        self.dependent_var_test = test_data[:, -1:]

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost = []


class LinearRegression(Regression):
    def __init__(
        self,
        independent_vars,
        dependent_var,
        iterations,
        learning_rate,
        train_split,
        seed,
        plot_style,
    ):
        """
        All inherited from Regression class
        """
        super().__init__(
            independent_vars,
            dependent_var,
            iterations,
            learning_rate,
            train_split,
            seed,
            plot_style,
        )

        # Add ones column to allow for beta 0
        self.independent_vars_train = np.c_[
            np.ones(len(self.independent_vars_train), dtype="int64"),
            self.independent_vars_train,
        ]
        self.independent_vars_test = np.c_[
            np.ones(len(self.independent_vars_test), dtype="int64"),
            self.independent_vars_test,
        ]

        # Initialize betas
        if len(self.independent_vars_train.shape) == 1:
            self.B = np.zeros(1)
        else:
            self.B = np.zeros(self.independent_vars_train.shape[1])

        # Automatically fit
        self.fit_gradient_descent()

    def predict(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict.dot(self.B).flatten()
        return predicted_values

    def find_gradient(self):
        estimate = self.predict(self.independent_vars_train)
        error = self.dependent_var_train.flatten() - estimate
        gradient = -(1.0 / len(self.independent_vars_train)) * error.dot(
            self.independent_vars_train
        )
        self.cost.append(np.power(error, 2))
        return gradient

    def fit_gradient_descent(self):
        for i in range(self.iterations):
            gradient = self.find_gradient()
            self.B = self.B - (self.learning_rate * gradient)

    def calculate_r_squared(self, independent_vars, dependent_var):
        sum_sq_r = np.sum((dependent_var - self.predict(independent_vars)) ** 2)
        sum_sq_t = np.sum((dependent_var - dependent_var.mean()) ** 2)
        return 1 - (sum_sq_r / sum_sq_t)

    def plot(self, increment=100,
             title=None, x_label=None, y_label=None):
        # If plot min / max not given, define based off of min and max of data
        for i in range(self.independent_vars_train.shape[1]-1):
            independent_var_train = self.independent_vars_train[:, i+1]
            independent_var_test = self.independent_vars_test[:, i+1]
            dependent_var_train = self.dependent_var_train[:, 0]
            dependent_var_test = self.dependent_var_test[:, 0]

            # # Set min/max axes
            min_independent_vars = min(np.min(independent_var_train), np.min(independent_var_test))
            max_independent_vars = max(np.max(independent_var_train), np.max(independent_var_test))
            min_dependent_vars = min(np.min(dependent_var_train), np.min(dependent_var_test))
            max_dependent_vars = max(np.max(dependent_var_train), np.max(dependent_var_test))

            # # Set x-axis range
            # independent_vars_hat = np.linspace(min_independent_vars, max_independent_vars, increment)
            # dependent_var_hat = self.predict(independent_vars_hat)

            plot_data_train = np.c_[independent_var_train, self.predict(self.independent_vars_train)]
            plot_data_train.sort(axis=0)

            # Plot regression line
            plt.plot(plot_data_train[:, 0], plot_data_train[:, 1], color='red')

            # Plot train points
            plt.scatter(independent_var_train, dependent_var_train, color='green')

            # Plot test points
            plt.scatter(independent_var_test, dependent_var_test, color='blue')

            plt.text(x=min_independent_vars * 1.1, y=max_dependent_vars * 0.9,
                     bbox=dict(),
                     s=f'')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.show()

    def __str__(self):
        return f"""
            Model Results
            -------------
            Betas: {[i for i in zip(range(len(self.B)), self.B)]}
            R^2 Train: {self.calculate_r_squared(self.independent_vars_train, self.dependent_var_train[:, 0])}
            R^2 Test: {self.calculate_r_squared(self.independent_vars_test, self.dependent_var_test[:, 0])}
            """
