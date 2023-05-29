import numpy as np
import matplotlib.pyplot as plt

def generate_training_set(num_samples):
    data = []
    for _ in range(num_samples):
        # Generate a random center point within the range (-1, 1) for both x and y coordinates
        center = np.random.uniform(-1, 1, size=2)
        # Generate a random radius within the range (0, 0.2)
        radius = np.random.uniform(0, 0.2)
        # Assign a random label of -1 or 1
        label = np.random.choice([-1, 1])
        # Create a dictionary representing the data point and append it to the data list
        data.append({'center': center, 'radius': radius, 'label': label})
    # Return the generated training data
    return data


def solve_optimization_problem(data, C, tolerance, max_iterations):
    num_samples = len(data)
    num_features = 2
    
    X = np.array([point['center'] for point in data])  # Extracts the center coordinates of each data point
    y = np.array([point['label'] for point in data])  # Extracts the labels of each data point
    deltas = np.array([point['radius'] for point in data])  # Extracts the radii of each data point
    
    # Initialize Lagrange multipliers (alpha) and threshold (b)
    alpha = np.zeros(num_samples)
    b = 0
    
    # Calculate Gram matrix
    K = np.dot(X, X.T)
    
    # Define kernel function
    def kernel(x1, x2):
        return np.dot(x1, x2)
    
    # Define prediction function
    def predict(x):
        wx = np.sum(alpha * y * kernel(X, x))  # Computes the weighted sum of the kernel evaluations
        return np.sign(wx + b)  # Returns the sign of the weighted sum
    
    # Perform SMO optimization
    iterations = 0
    while iterations < max_iterations:
        num_changed_alphas = 0
        for i in range(num_samples):
            E_i = predict(X[i]) - y[i]  # Computes the prediction error for the current data point
            
            # Check if the KKT conditions are violated for the current data point
            if (y[i] * E_i < -tolerance and alpha[i] < C) or (y[i] * E_i > tolerance and alpha[i] > 0):
                j = np.random.choice([k for k in range(num_samples) if k != i])  # Randomly choose another data point index
                
                E_j = predict(X[j]) - y[j]  # Computes the prediction error for the randomly chosen data point
                
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                
                # Calculate the bounds for the Lagrange multiplier
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                # Skip to the next iteration if the bounds are equal
                if L == H:
                    continue
                
                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])  # Compute the second-order term of the objective function
                if eta >= 0:
                    continue
                
                # Update alpha[j] and clip it within the bounds
                alpha[j] -= y[j] * (E_i - E_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                # Skip to the next iteration if the update is too small
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha[i] based on the change in alpha[j]
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                
                b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[i]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[i], X[j])
                b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[j]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[j], X[j])
                
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                num_changed_alphas += 1
        
        if num_changed_alphas == 0:
            iterations += 1
        else:
            iterations = 0
    
    # Select support vectors
    support_vectors = []
    for i in range(num_samples):
        if alpha[i] > 0:
                    support_vectors.append({'center': X[i], 'label': y[i], 'radius': deltas[i]})
    
    # Compute weight vector
    w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)
    
    # Compute bias
    b = np.mean([y[i] - np.sum(alpha * y * kernel(X, X[i])) for i in range(num_samples) if alpha[i] > 0])
    
    return support_vectors, w, b



def compute_weight(alpha_star, data):
    w = np.zeros(2)
    for i, point in enumerate(data):
        w += alpha_star[i] * point['label'] * point['center']
    return w

def compute_bias(alpha_star, data):
    for i, point in enumerate(data):
        if 0 < alpha_star[i] < C:
            j = i
            break
    b = point['label'] - np.sum([alpha_star[k] * data[k]['label'] * np.dot(point['center'], data[k]['center']) for k in range(len(data))])
    return b

def visualize_separator(data, w, b):
    plt.figure(figsize=(8, 8))
    
    for point in data:
        center = point['center']
        radius = point['radius']
        label = point['label']
        if label == 1:
            plt.scatter(center[0], center[1], color='blue', marker='o')
        else:
            plt.scatter(center[0], center[1], color='red', marker='o')
        circle = plt.Circle(center, radius, fill=False)
        plt.gca().add_patch(circle)
    
    x = np.linspace(-1, 1, 100)
    y = (-w[0] * x - b) / w[1]
    plt.plot(x, y, color='black')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
def calculate_train_score(data, w, b):
    num_correct = 0
    for point in data:
        x = point['center']
        label = point['label']
        prediction = np.sign(np.dot(w, x) + b)
        if prediction == label:
            num_correct += 1
    score = num_correct / len(data)
    return score


# Step 1: Generate Training Set
num_samples = 50
data = generate_training_set(num_samples)

# Step 2: Choose Penalty Parameter C
C = 1.0

# Step 3: Solve the Optimization Problem
tolerance = 0.01
max_iterations = 1000
support_vectors, w, b = solve_optimization_problem(data, C, tolerance, max_iterations)

# Step 4: Compute Training Score
train_score = calculate_train_score(data, w, b)
print("Training Score:", train_score)

# Step 5: Visualize the Separator
visualize_separator(data, w, b)

