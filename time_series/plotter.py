import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class LinearRegressionPlotter:
    def __init__(self):
        """
        Initializes the LinearRegressionPlotter class with an empty linear model.
        """
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fits the linear regression model using the provided training data.

        Parameters:
        - X: Features (independent variables), numpy array or similar format, shape (n_samples, n_features)
        - y: Target (dependent variable), numpy array or similar format, shape (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Makes predictions using the linear regression model on the provided data.

        Parameters:
        - X: Data for which predictions are to be made, numpy array or similar format, shape (n_samples, n_features)

        Returns:
        - Predicted values, numpy array, shape (n_samples,)
        """
        return self.model.predict(X)

    def plot_data_and_line(self, X_train, y_train, X_test=None, y_test=None, y_pred=None, show=True):
        """
        Plots the training data, optional testing data, and the linear regression line.

        Parameters:
        - X_train: Training features, numpy array, shape (n_samples, n_features)
        - y_train: Training targets, numpy array, shape (n_samples,)
        - X_test: Optional, testing features, numpy array, shape (n_samples, n_features)
        - y_test: Optional, testing targets, numpy array, shape (n_samples,)
        - y_pred: Optional, predictions for the testing data, numpy array, shape (n_samples,)
        - show: Boolean, if True, display the plot; otherwise, do not display.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training data')

        if X_test is not None and y_test is not None:
            plt.scatter(X_test, y_test, color='red', label='Testing data')

        if X_test is not None and y_pred is not None:
            plt.plot(X_test, y_pred, color='green', label='Linear regression line')

        plt.title('Linear Regression Plot')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        
        if show:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Dummy data
    X = np.random.rand(100, 1) * 10  # 100 random X values
    y = 2 * X + 1 + np.random.randn(100, 1) * 2  # Corresponding y values with some noise

    # Splitting the data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create an instance of the class
    plotter = LinearRegressionPlotter()

    # Fit the model
    plotter.fit(X_train, y_train)

    # Predict using the model
    y_pred = plotter.predict(X_test)

    # Plot data and regression line
    plotter.plot_data_and_line(X_train, y_train, X_test, y_test, y_pred, show=True)
