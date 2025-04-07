import data_generator as dg
import numpy as np
import matplotlib.pyplot as plt
import math

def design_matrix(x, n):
    return np.array([x**i for i in range(n)]).reshape(-1, 1)

def draw_ground_truth(W, a, n, title="Ground truth", save_path=None):
    x_plot = np.linspace(-2, 2, 200)
    y_true = []
    y_upper = []
    y_lower = []
    for x in x_plot:
        phi = design_matrix(x, n)
        y = float(np.dot(W, phi.flatten()))  # y = w^T phi(x)
        y_true.append(y)
        std = math.sqrt(1 / a)
        y_upper.append(y + std)
        y_lower.append(y - std)

    plt.figure(figsize=(5, 4))
    plt.plot(x_plot, y_true, 'k-', label='Ground Truth')
    plt.plot(x_plot, y_upper, 'r-', linewidth=1)
    plt.plot(x_plot, y_lower, 'r-', linewidth=1)
    #plt.fill_between(x_plot, y_lower, y_upper, color='red', alpha=0.2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-12, 25)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def draw_plot(mean, variance, seen_x, seen_y, title, save_path=None):
    x_plot = np.linspace(-2, 2, 200)
    y_mean = []
    y_std = []

    for x_star in x_plot:
        phi_star = design_matrix(x_star, len(mean))
        mu = (phi_star.T @ mean).item()
        var = (phi_star.T @ variance @ phi_star + 1 / a).item()
        y_mean.append(mu)
        y_std.append(np.sqrt(var))

    plt.figure(figsize=(5, 4))

    # 黑線 = 預測 mean
    plt.plot(x_plot, y_mean, 'k-', linewidth=2)

    # 紅線 = 上下 1 std
    plt.plot(x_plot, np.array(y_mean) + np.array(y_std), 'r-', linewidth=1)
    plt.plot(x_plot, np.array(y_mean) - np.array(y_std), 'r-', linewidth=1)

    # 紅色區域 = 不確定性區間
    #plt.fill_between(x_plot,
    #                 np.array(y_mean) - np.array(y_std),
    #                 np.array(y_mean) + np.array(y_std),
    #                 color='red', alpha=0.2)

    # 藍點 = 觀察到的資料
    if seen_x and seen_y:
        plt.scatter(seen_x, seen_y, color='blue', s=5)

    # 標題與格式
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-12, 25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def draw_combined_plots(W, a, n, mean, variance, mean_10, var_10, mean_50, var_50, seen10_x, seen10_y, seen50_x, seen50_y, seen_x, seen_y):
    x_plot = np.linspace(-2, 2, 200)
    y_true = []
    y_upper = []
    y_lower = []
    for x in x_plot:
        phi = design_matrix(x, n)
        y = float(np.dot(W, phi.flatten()))  # y = w^T phi(x)
        y_true.append(y)
        var = a
        y_upper.append(y + var)
        y_lower.append(y - var)
    def predict(mean, variance):
        y_mean = []
        y_std = []
        for x_star in x_plot:
            phi_star = design_matrix(x_star, len(mean))
            mu = (phi_star.T @ mean).item()
            var = (phi_star.T @ variance @ phi_star + a).item()
            y_mean.append(mu)
            y_std.append(var)
        return y_mean, y_std

    y_mean_10, y_std_10 = predict(mean_10, var_10)
    y_mean_50, y_std_50 = predict(mean_50, var_50)
    y_mean, y_std = predict(mean, variance)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: Ground truth
    axs[0, 0].plot(x_plot, y_true, 'k-')
    axs[0, 0].plot(x_plot, y_upper, 'r-')
    axs[0, 0].plot(x_plot, y_lower, 'r-')
    axs[0, 0].set_title("Ground truth")
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 0].set_ylim(-12, 25)

    # Top-right: Predict result
    axs[0, 1].plot(x_plot, y_mean, 'k-')
    axs[0, 1].plot(x_plot, np.array(y_mean) + np.array(y_std), 'r-')
    axs[0, 1].plot(x_plot, np.array(y_mean) - np.array(y_std), 'r-')
    axs[0, 1].scatter(seen_x, seen_y, color='blue', s=5)
    axs[0, 1].set_title("Predict result")
    axs[0, 1].set_xlim(-2, 2)
    axs[0, 1].set_ylim(-12, 25)

    # Bottom-left: After 10 incomes
    axs[1, 0].plot(x_plot, y_mean_10, 'k-', linewidth=2)
    axs[1, 0].plot(x_plot, np.array(y_mean_10) + np.array(y_std_10), 'r-')
    axs[1, 0].plot(x_plot, np.array(y_mean_10) - np.array(y_std_10), 'r-')
    axs[1, 0].scatter(seen10_x, seen10_y, color='blue', s=5)
    axs[1, 0].set_title("After 10 incomes")
    axs[1, 0].set_xlim(-2, 2)
    axs[1, 0].set_ylim(-12, 25)

    # Bottom-right: After 50 incomes
    axs[1, 1].plot(x_plot, y_mean_50, 'k-')
    np.array(y_mean) + np.array(y_std)
    axs[1, 1].plot(x_plot, np.array(y_mean_50) + np.array(y_std_50), 'r-')
    axs[1, 1].plot(x_plot, np.array(y_mean_50) - np.array(y_std_50), 'r-')
    axs[1, 1].scatter(seen50_x, seen50_y, color='blue', s=5)
    axs[1, 1].set_title("After 50 incomes")
    axs[1, 1].set_xlim(-2, 2)
    axs[1, 1].set_ylim(-12, 25)

    plt.tight_layout()
    #plt.savefig("result_plot/case3.png")
    plt.show()

b = float(input("Precision of Prior (b): "))
n = int(input("Basis Number: "))
a = float(input("Precision of Likelihood (a): "))
w = input("W：")
W = [float(x) for x in w.split()]
#print(f"Design matrix: {phy}")
#draw_ground_truth(W, a, n, title="Ground truth", save_path="gt.png")

# Initialize prior mean and variance
mean = np.zeros(n)
variance = np.eye(n) / b
print(f"Initial mean: {mean}, Initial variance: {variance}")

count = 0
epsilon = 1e-5
seen_x = []
seen_y = []
while True:
    mean_prev = mean.copy()
    var_prev = variance.copy()
    x, y = dg.linear_model_generator(n, W, a)
    seen_x.append(x)
    seen_y.append(y)
    print(f"Add data point ({x}, {y}): ")
    print()
    phi = design_matrix(x, n)
    #print(f"Design matrix: {phi}")

    # Update posterior covariance
    variance_inv = np.linalg.inv(variance)
    variance = np.linalg.inv(variance_inv + (1/a) * phi @ phi.T)

    # Update posterior mean
    mean = variance @ (variance_inv @ mean + (1/a) * phi.flatten() * y)

    print(f"Posterior mean:")
    print(mean)
    print()
    print(f"Posterior variance:")
    print(variance)
    print()
    phi_star = phi
    predictive_mean = float(phi_star.T @ mean)
    predictive_var = float(phi_star.T @ variance @ phi_star + a)
    print(f"Predictive distribution ~ N({predictive_mean:.5f}, {predictive_var:.5f})")
    print()
    count += 1
    if count == 10:
        seen10_x = seen_x.copy()
        seen10_y = seen_y.copy()
        mean_10 = mean.copy()
        var_10 = variance.copy()
        #draw_plot(mean, variance, seen_x, seen_y, "After 10 incomes", None)
    elif count == 50:
        seen50_x = seen_x.copy()
        seen50_y = seen_y.copy()
        mean_50 = mean.copy()
        var_50 = variance.copy()
        #draw_plot(mean, variance, seen_x, seen_y, "After 50 incomes", None)
    if np.all(np.abs(mean - mean_prev) < epsilon) and np.all(np.abs(variance - var_prev) < epsilon):
        print("Posterior has converged.")
        #draw_plot(mean, variance, seen_x, seen_y, "predict result", "predict_result.png")
        draw_combined_plots(W, a, n, mean, variance, mean_10, var_10, mean_50, var_50, seen10_x, seen10_y, seen50_x, seen50_y, seen_x, seen_y)
        break
    