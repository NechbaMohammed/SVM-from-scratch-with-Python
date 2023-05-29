import numpy as np
import matplotlib.pyplot as plt

def polynomial_kernel(X1, X2, degree=2):
    return (np.dot(X1, X2) + 1) ** degree

def rbf_kernel(X1, X2, sigma=1.0):
    return np.exp(-np.linalg.norm(X1 - X2) ** 2 / (2 * sigma ** 2))

def smo(X, y, kernel, epsilon, C, max_iter=200, tol=1e-3):
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0
    kernel_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = kernel(X[i], X[j])
    iter_count = 0
    while iter_count < max_iter:
        alpha_prev = np.copy(alpha)
        for i in range(n_samples):
            error_i = np.sum(alpha * y * kernel_matrix[:, i]) - y[i] + b
            if (y[i] * error_i < -epsilon and alpha[i] < C) or (y[i] * error_i > epsilon and alpha[i] > 0):
                j = np.random.choice(np.concatenate((np.arange(i), np.arange(i+1, n_samples))), size=1)[0]
                error_j = np.sum(alpha * y * kernel_matrix[:, j]) - y[j] + b
                alpha_i_prev = alpha[i]
                alpha_j_prev = alpha[j]
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue
                eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                if eta >= 0:
                    continue
                alpha[j] = alpha[j] - (y[j] * (error_i - error_j)) / eta
                alpha[j] = min(H, alpha[j])
                alpha[j] = max(L, alpha[j])
                if abs(alpha[j] - alpha_j_prev) < tol:
                    continue
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_prev - alpha[j])
                b1 = b - error_i - y[i] * (alpha[i] - alpha_i_prev) * kernel_matrix[i, i] - y[j] * (alpha[j] - alpha_j_prev) * kernel_matrix[i, j]
                b2 = b - error_j - y[i] * (alpha[i] - alpha_i_prev) * kernel_matrix[i, j] - y[j] * (alpha[j] - alpha_j_prev) * kernel_matrix[j, j]
                if 0 < alpha[i] and alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
        if np.linalg.norm(alpha - alpha_prev) < tol:
            break
        iter_count += 1
    support_vectors = np.where(alpha > 0)[0]
    return alpha,b

def epsilon_band_svrnr(X_train, y_train, X_test, kernel, degree=2, sigma=1.0, epsilon=0.1, C=1.0,max_iter=200):
    alpha, b = smo(X_train, y_train, kernel, epsilon,max_iter, C)
    y_pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        prediction = 0
        for j in range(X_train.shape[0]):
            prediction += alpha[j] * y_train[j] * kernel(X_train[j], X_test[i])
        y_pred[i] = prediction + b
    return y_pred

# Step 1: Choose a simple nonlinear dataset
X = np.array([-5.        , -4.79591837, -4.59183673, -4.3877551 , -4.18367347,
       -3.97959184, -3.7755102 , -3.57142857, -3.36734694, -3.16326531,
       -2.95918367, -2.75510204, -2.55102041, -2.34693878, -2.14285714,
       -1.93877551, -1.73469388, -1.53061224, -1.32653061, -1.12244898,
       -0.91836735, -0.71428571, -0.51020408, -0.30612245, -0.10204082,
        0.10204082,  0.30612245,  0.51020408,  0.71428571,  0.91836735,
        1.12244898,  1.32653061,  1.53061224,  1.73469388,  1.93877551,
        2.14285714,  2.34693878,  2.55102041,  2.75510204,  2.95918367,
        3.16326531,  3.36734694,  3.57142857,  3.7755102 ,  3.97959184,
        4.18367347,  4.3877551 ,  4.59183673,  4.79591837,  5.        ])
y = np.sin(X) + 0.3 * X + np.random.normal(0, 0.2, len(X))

# Step 2: Exploratory analysis and preprocessing (not required for this example)

# Step 3: Split data into train and test
X_train = X[:50]
y_train = y[:50]
X_test = X[1:]
y_test = y[1:]

# Step 5: Implement epsilon-band SVRNR with kernels
y_pred_poly = epsilon_band_svrnr(X_train, y_train, X_test, polynomial_kernel, degree=2,max_iter=1000,C=0.5)
y_pred_rbf = epsilon_band_svrnr(X_train, y_train, X_test, rbf_kernel, sigma=3.0,max_iter=1000)

# Step 6: Train the models and compare MSE
mse_poly = np.mean((y_pred_poly - y_test) ** 2)
mse_rbf = np.mean((y_pred_rbf - y_test) ** 2)
print("MSE (Polynomial Kernel):", mse_poly)
print("MSE (RBF Kernel):", mse_rbf)

# Step 7: Visualize the regression results
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred_rbf, label="RBF Kernel")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred_poly, label="Polynomial Kernel")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


