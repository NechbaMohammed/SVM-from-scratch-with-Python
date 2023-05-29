import numpy as np
import matplotlib.pyplot as plt

# Generate the data points
np.random.seed(0)

m = 10  # Number of positive samples
n = 30  # Number of negative samples for each class

X_pos = np.random.normal(loc=[5, 3], scale=[2, 1], size=(m, 2))
X_neg = np.random.normal(loc=[5, 2], scale=[2, 1], size=(n, 2))

X = np.vstack((X_pos, X_neg))
y = np.concatenate((np.ones(m), -np.ones(n)))

# Compute the mean of each class
mean_pos = np.mean(X_pos, axis=0)
mean_neg = np.mean(X_neg, axis=0)

# Compute the between-class scatter matrix
mean_diff = mean_pos - mean_neg
Sb = np.outer(mean_diff, mean_diff)

# Compute the within-class scatter matrix
Sw = np.zeros((2, 2))
for i in range(m):
    diff = X_pos[i] - mean_pos
    Sw += np.outer(diff, diff)
for i in range(n):
    diff = X_neg[i] - mean_neg
    Sw += np.outer(diff, diff)

# Solve the eigenvalue problem
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indices]
eigen_vectors = eigen_vectors[:, sorted_indices]

# Choose the eigenvector corresponding to the largest eigenvalue
w = eigen_vectors[:, 0]

# Normalize w to have unit length
w /= np.linalg.norm(w)

# Compute the bias term b
b = -0.5 * (mean_pos + mean_neg) @ w

# Separating hyperplane function
def separating_hyperplane(x):
    return w @ x + b

# Plotting the data points
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Positive')
plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Negative')

# Plotting the separating hyperplane
x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
x_plot = np.linspace(x_min, x_max, 100)
y_plot = -(w[0] / w[1]) * x_plot - (b / w[1])
plt.plot(x_plot, y_plot, color='black', linestyle='dashed', label='Separating Hyperplane')

# Setting up the plot
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data Points and Separating Hyperplane')
plt.legend()

# Display the plot
plt.show()
