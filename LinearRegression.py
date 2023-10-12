import numpy as np

class LinearRegression:
    def __init__(self):
        self.intercept_ = None  # Intercept of the linear model
        self.coef_ = None       # Coefficient (slope) of the linear model

    def fit_analytic(self, X, y):
        """
        Fit the linear regression model using the analytic (closed-form) solution.
        
        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        y (array-like): Target values of shape (n_samples,).

        Returns:
        self (LinearRegression): Fitted linear regression model.
        """
        # Calculate the coefficients using the analytic solution
        X = np.array(X)
        y = np.array(y)
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        self.coef_ = numerator / denominator
        self.intercept_ = y_mean - self.coef_ * X_mean
        
        return self

    def fit_numerical(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Fit the linear regression model using the numerical solution (gradient descent).
        
        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        y (array-like): Target values of shape (n_samples,).
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for gradient descent.

        Returns:
        self (LinearRegression): Fitted linear regression model.
        """
        # Initialize coefficients
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(n_iterations):
            # Compute predictions
            y_pred = self.predict(X)

            # Compute gradients
            grad_coef = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            grad_intercept = -(2 / n_samples) * np.sum(y - y_pred)

            # Update coefficients using gradient descent
            self.coef_ -= learning_rate * grad_coef
            self.intercept_ -= learning_rate * grad_intercept

        return self

    def predict(self, X):
        """
        Make predictions using the linear regression model.
        
        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
        y_pred (array-like): Predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.coef_) + self.intercept_

    def r_squared(self, X, y):
        """
        Calculate the R-squared (coefficient of determination) of the model.
        
        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        y (array-like): True target values of shape (n_samples,).

        Returns:
        r2 (float): R-squared value.
        """
        y_pred = self.predict(X)
        ssr = np.sum((y_pred - np.mean(y))**2)
        sst = np.sum((y - np.mean(y))**2)
        r2 = ssr / sst
        return r2

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])

# Initialize and fit the model using the analytic solution
model_analytic = LinearRegression()
model_analytic.fit_analytic(X, y)

# Initialize and fit the model using the numerical solution
model_numerical = LinearRegression()
model_numerical.fit_numerical(X, y)

# Make predictions
X_new = np.array([6, 7, 8])
y_pred_analytic = model_analytic.predict(X_new)
y_pred_numerical = model_numerical.predict(X_new)

# Calculate R-squared
r2_analytic = model_analytic.r_squared(X, y)
r2_numerical = model_numerical.r_squared(X, y)

print("Analytic Solution - Coefficients:", model_analytic.coef_, model_analytic.intercept_)
print("Analytic Solution - Predictions:", y_pred_analytic)
print("Analytic Solution - R-squared:", r2_analytic)

print("Numerical Solution - Coefficients:", model_numerical.coef_, model_numerical.intercept_)
print("Numerical Solution - Predictions:", y_pred_numerical)
print("Numerical Solution - R-squared:", r2_numerical)