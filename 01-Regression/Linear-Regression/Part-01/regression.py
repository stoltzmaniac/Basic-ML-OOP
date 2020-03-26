# All models for modeling
import numpy as np


class Regression:
    def __init__(
        self,
        independent_vars: np.ndarray,
        dependent_var: np.ndarray,
        iterations: int,
        learning_rate: float,
        train_split: float,
        seed: int,
    ):
        """
        :param independent_vars: np.ndarray
        :param dependent_var: np.array (one dimensional)
        :param iterations: int
        :param learning_rate: float
        :param train_split: float (0 < value < 1)
        :param seed: int
        """
        # Check data types
        if type(seed) != int:
            raise ValueError(f"seed value not an int")
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

    def plot(self):
        pass


class LinearRegression(Regression):
    def __init__(
        self,
        independent_vars,
        dependent_var,
        iterations,
        learning_rate,
        train_split,
        seed,
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

    def __str__(self):
        return f"""
            Model Results
            -------------
            Betas: {[i for i in zip(range(len(self.B)), self.B)]}
            R^2 Train: {self.calculate_r_squared(self.independent_vars_train, self.dependent_var_train[:, 0])}
            R^2 Test: {self.calculate_r_squared(self.independent_vars_test, self.dependent_var_test[:, 0])}
            """
