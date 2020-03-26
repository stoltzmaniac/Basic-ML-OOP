import numpy as np
import matplotlib.pyplot as plt


class Regression:
    def __init__(
        self,
        predictor_vars: np.ndarray,
        response_var: np.ndarray,
        iterations: int,
        learning_rate: float,
        train_split: float,
        seed: int,
    ):
        """
        :param predictor_vars: np.ndarray
        :param response_var: np.array (one dimensional)
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
            type(predictor_vars) == np.ndarray,
            type(response_var) == np.ndarray,
        ]
        if not all(type_check_arrays):
            raise ValueError(f"Type(s) of data for Regression class are not accurate")
        if (
            not type(learning_rate) == float
            and type(train_split) == float
            and train_split <= 1
        ):
            raise ValueError(f"learning_rate or train_split not acceptable input(s)")
        if response_var.shape[0] != predictor_vars.shape[0]:
            raise ValueError(
                f"Dimension(s) of data for Regression class are not accurate"
            )

        all_data = np.column_stack((predictor_vars, response_var))
        all_data = all_data.astype("float")
        np.random.seed(seed)
        np.random.shuffle(all_data)

        split_row = round(all_data.shape[0] * train_split)
        train_data = all_data[:split_row]
        test_data = all_data[split_row:]

        # Train
        self.predictor_vars_train = train_data[:, :-1]
        self.response_var_train = train_data[:, -1:]

        # Test
        self.predictor_vars_test = test_data[:, :-1]
        self.response_var_test = test_data[:, -1:]

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost = []

    def plot(self):
        pass


class LinearRegression(Regression):
    def __init__(
        self,
        predictor_vars,
        response_var,
        iterations,
        learning_rate,
        train_split,
        seed,
        tolerance,
        batch_size,
        max_epochs,
        decay,
    ):
        """
        All inherited from Regression class
        """
        super().__init__(
            predictor_vars,
            response_var,
            iterations,
            learning_rate,
            train_split,
            seed,
        )

        # Add ones column to allow for beta 0
        self.predictor_vars_train = np.c_[
            np.ones(len(self.predictor_vars_train), dtype="int64"),
            self.predictor_vars_train,
        ]

        self.predictor_vars_test = np.c_[
            np.ones(len(self.predictor_vars_test), dtype="int64"),
            self.predictor_vars_test,
        ]

        # Initialize betas
        if len(self.predictor_vars_train.shape) == 1:
            self.B = np.random.randn(1)
        else:
            self.B = np.random.randn(self.predictor_vars_train.shape[1])

        self.max_epochs = max_epochs
        self.decay = decay
        self.tolerance = tolerance
        self.batch_size = batch_size

        # Automatically fit
        #self.fit_gradient_descent()

    def predict(self, values_to_predict: np.ndarray):
        data = np.c_[np.ones(len(values_to_predict), dtype="int64"),
              values_to_predict]
        predicted_values = data.dot(self.B).flatten()
        return predicted_values

    def predict_(self, values_to_predict: np.ndarray):
        predicted_values = values_to_predict.dot(self.B).flatten()
        return predicted_values

    def find_gradient(self, x, y):
        estimate = self.predict_(values_to_predict=x)
        error = (y.flatten() - estimate)
        mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
        gradient = -(1.0 / len(x)) * error.dot(x)
        return {'gradient': gradient, 'error': mse}

    def fit_stochastic_gradient_descent(self):
        epoch = 0
        error = 1
        while True:
            order = np.random.permutation(len(self.predictor_vars_train))
            self.predictor_vars_train = self.predictor_vars_train[order]
            self.response_var_train = self.response_var_train[order]
            b = 0
            while b < len(self.predictor_vars_train):
                tx = self.predictor_vars_train[b: b + self.batch_size]
                ty = self.response_var_train[b: b + self.batch_size]
                gradient_data = self.find_gradient(x=tx, y=ty)
                gradient = gradient_data['gradient']
                error = gradient_data['error']
                self.B -= self.learning_rate * gradient
                b += self.batch_size

            if epoch % 100 == 0:
                epoch_gradient = self.find_gradient(self.predictor_vars_train, self.response_var_train)
                print(f"Epoch: {epoch} - Error: {epoch_gradient['error']}")
                print(abs(error - epoch_gradient['error']))
                if abs(error - epoch_gradient['error']) < self.tolerance:
                    print('Converged')
                    break
                if epoch >= self.max_epochs:
                    print('Max Epochs limit reached')
                    break

            epoch += 1
            self.learning_rate = self.learning_rate * (self.decay ** int(epoch / 1000))

    def calculate_r_squared(self, predictor_vars, response_var):
        sum_sq_r = np.sum((response_var - self.predict_(predictor_vars)) ** 2)
        sum_sq_t = np.sum((response_var - response_var.mean()) ** 2)
        return 1 - (sum_sq_r / sum_sq_t)

    def __str__(self):
        return f"""
            Model Results
            -------------
            Betas: {[i for i in zip(range(len(self.B)), self.B)]}
            R^2 Train: {self.calculate_r_squared(self.predictor_vars_train, self.response_var_train[:, 0])}
            R^2 Test: {self.calculate_r_squared(self.predictor_vars_test, self.response_var_test[:, 0])}
            """
