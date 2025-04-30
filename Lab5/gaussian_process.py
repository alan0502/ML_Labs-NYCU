import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def rational_quadratic_kernel(x1, x2, alpha, length_scale, sigma_f):
    d = np.sum((x1[:, None] - x2[None, :])**2, axis=-1)
    return (sigma_f ** 2) * (1 + d / (2 * alpha * length_scale ** 2)) ** (-alpha)

def predict_kernel(X_train, y_train, X_test, alpha, length_scale, sigma_f):
    beta = 5
    k = rational_quadratic_kernel(X_train, X_train, alpha, length_scale, sigma_f)
    k_star = rational_quadratic_kernel(X_train, X_test, alpha, length_scale, sigma_f)
    k_star_star = rational_quadratic_kernel(X_test, X_test, alpha, length_scale, sigma_f)
    beta_I = np.eye(len(X_train)) * (1/beta)
    k += beta_I
    beta_I = np.eye(len(X_test)) * (1/beta)
    k_star_star += beta_I
    k_inv = np.linalg.inv(k)
    mu = k_star.T @ k_inv @ y_train
    cov = k_star_star - k_star.T @ k_inv @ k_star
    return mu, np.sqrt(np.diag(cov))

def marginal_likelihood(parameters, X_train, y_train):
    beta = 5
    log_alpha, log_length_scale, log_sigma_f = parameters
    alpha = np.exp(log_alpha)
    length_scale = np.exp(log_length_scale)
    sigma_f = np.exp(log_sigma_f)
    k = rational_quadratic_kernel(X_train, X_train, alpha, length_scale, sigma_f)
    beta_I = np.eye(len(X_train)) * (1/beta)
    k += beta_I
    k_inv = np.linalg.inv(k)
    return -(- len(X_train) / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(k)) - 0.5 * y_train.T @ k_inv @ y_train)

data = np.loadtxt('data/input.data')
X_train = data[:, 0].reshape(-1, 1)
y_train = data[:, 1]
#print(X_train)
X_min, X_max = X_train.min(), X_train.max()
#print(X_min, X_max)
X_test = np.linspace(X_min - 1, X_max + 1, 100).reshape(-1, 1)

parameters = np.log([1.0, 1.0, 1.0])
res = minimize(marginal_likelihood, parameters, args=(X_train, y_train), method='L-BFGS-B')
final_alpha, final_length_scale, final_sigma_f = np.exp(res.x)
mean, std = predict_kernel(X_train, y_train, X_test, final_alpha, final_length_scale, final_sigma_f)
#print(X.shape, y.shape)
#print(X[:5], y[:5])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'kx', label='Training Data')  # black x for training points
plt.plot(X_test, mean, 'r-', label='Predicted Mean')      # blue line for mean
plt.plot(X_test, mean - 2 * std, 'r--', linewidth=0.5)
plt.plot(X_test, mean + 2 * std, 'r--', linewidth=0.5)  # blue line for mean ± 2 std
plt.fill_between(X_test.flatten(), 
                 mean - 2 * std, 
                 mean + 2 * std, 
                 color="red", 
                 #label='95% Confidence Interval'
                 alpha=0.2
                 )
plt.title('Gaussian Process Regression with Rational Quadratic Kernel')
plt.legend()
plt.grid(True)
plt.savefig('result/task2.png', dpi=300, bbox_inches='tight')
plt.show()