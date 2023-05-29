import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Define the random sampling function
def sample_uniform(a, b, size):
    return np.random.uniform(a, b, size)

# Define the optimization problem solver
def optimize(X, y, p):
    n, d = X.shape

    # Define the hinge loss function
    def hinge_loss(theta):
        w, b = theta[:-1], theta[-1]
        pred = X.dot(w) + b
        return np.maximum(0, 1 - y * pred)

    # Define the objective function
    def objective(theta):
        w, b = theta[:-1], theta[-1]
        return np.linalg.norm(w) + C * np.sum(p * hinge_loss(theta))

    # Initialize the weights and bias
    theta0 = np.ones(d+1)

    # Define the equality constraint
    def equality_constraint(theta):
        return np.dot(y, X.dot(theta[:-1]) + theta[-1]) - n

    # Define the optimization problem
    constraints = {'type': 'eq', 'fun': equality_constraint}
    bounds = [(None, None)] * d + [(None, None)]
    result = minimize(objective, theta0, method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract the optimized weights and bias
    w = result.x[:-1]
    b = result.x[-1]

    return w, b

def plot_data_and_hyperplane(X, y, w, b):
    # Separate positive and negative samples
    positive_samples = X[y == 1]
    negative_samples = X[y == -1]

    # Plot the data points
    plt.scatter(positive_samples[:, 0], positive_samples[:, 1], color='blue', label='Positive')
    plt.scatter(negative_samples[:, 0], negative_samples[:, 1], color='red', label='Negative')

    # Plot the separating hyperplane
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, color='black', label='Separating Hyperplane')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Data and Separating Hyperplane')
    plt.grid(True)
    plt.show()
    
# Set the parameters
C = 200
a = 2
p_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Generate the data
n_values = [20, 50, 100]
np.random.seed(42)
data = []
for n in n_values:
    m = n // 2  # Half of the data points will have label 1, and half will have label -1
    X = np.zeros((n, 2))
    y = np.concatenate((np.ones(m), -np.ones(n - m)))
    for i in range(n):
        if i < m:
            X[i] = [sample_uniform(1, 2, 1), sample_uniform(2, 3, 1)]
        else:
            X[i] = [sample_uniform(2, 3, 1), sample_uniform(1, 2, 1)]
    data.append((X, y))

# Solve the optimization problem for each dataset
for i, (X, y) in enumerate(data):
    print(f"Dataset {i+1}:")
    for p in p_values:
        w, b = optimize(X, y, p)
        

plot_data_and_hyperplane(X, y, w, b)

