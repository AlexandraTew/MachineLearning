import numpy as np

class LogisticRegressionGD:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, max_iters=10000, epsilon=1e-5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon

        # Initialize weights and biases
        self.theta1 = np.random.rand(hidden_size, input_size + 1)
        self.theta2 = np.random.rand(output_size, hidden_size + 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        # Add bias to input
        a1 = np.insert(X, 0, 1, axis=1)

        # Hidden layer
        z2 = np.dot(a1, self.theta1.T)
        a2 = self.sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis=1)  # Add bias

        # Output layer
        z3 = np.dot(a2, self.theta2.T)
        a3 = self.sigmoid(z3)

        return a1, z2, a2, z3, a3

    def compute_cost(self, y, a3):
        m = len(y)
        cost = (-1 / m) * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
        return cost

    def backward_propagation(self, a1, z2, a2, z3, a3, y):
        m = len(y)

        # Output layer
        delta3 = a3 - y
        grad_theta2 = (1 / m) * np.dot(delta3.T, a2)

        # Hidden layer
        delta2 = np.dot(delta3, self.theta2[:, 1:]) * (a2[:, 1:] * (1 - a2[:, 1:]))
        grad_theta1 = (1 / m) * np.dot(delta2.T, a1)

        return grad_theta1, grad_theta2

    def update_parameters(self, grad_theta1, grad_theta2):
        self.theta1 -= self.learning_rate * grad_theta1
        self.theta2 -= self.learning_rate * grad_theta2

    def fit(self, X, y):
        for i in range(self.max_iters):
            # Forward propagation
            a1, z2, a2, z3, a3 = self.forward_propagation(X)

            # Calculate cost
            cost = self.compute_cost(y, a3)

            # Backward propagation
            grad_theta1, grad_theta2 = self.backward_propagation(a1, z2, a2, z3, a3, y)

            # Update parameters
            self.update_parameters(grad_theta1, grad_theta2)

            # Stopping criterion
            if cost < self.epsilon:
                print(f"Converged after {i+1} iterations.")
                break

            # Print cost every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}, Cost: {cost}")

    def predict(self, X):
        _, _, _, _, a3 = self.forward_propagation(X)
        predictions = (a3 >= 0.5).astype(int)
        return predictions

# Example usage with Iris dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Setosa vs. non-Setosa

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias to input
X = np.insert(X, 0, 1, axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegressionGD(input_size=4, hidden_size=2, output_size=1)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")