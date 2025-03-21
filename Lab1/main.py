import numpy as np
import LSE as lse
import Steepest_descent as std
import Newton_method as newton
import matplotlib.pyplot as plt

def read_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                x, y = map(float, line.strip().split(","))
                data.append((x, y))
            except ValueError:
                print(f"error -> {line.strip()}")
    
    data = np.array(data)
    X, y = data[:, 0], data[:, 1]
    return X, y

def design_matrix(X, degree):
    return np.vander(X, degree, increasing=True)

def cal_fittingline(w):
    equation = []
    for i in range(0, len(w)):
        if i == 0:
            equation.append(f"{w[i]}")
        else:
            equation.append(f"{w[i]} x^{i}")
    #print(equation)
    eq = " + ".join(equation)
    print(f"Fitting line: y = {eq}")

def cal_error(A, w, b):
    total_error = 0
    pred = A @ w
    for i in range(len(b)):
        total_error += (pred[i] - b[i])**2
    print(f"Total error: {total_error}")

N = int(input("N: "))
lam = int(input("lambda: "))
file_path = "inputfile.txt"
X, y = read_data(file_path)
print(X)
print(y)
A = design_matrix(X, N)

# LSE
res_lse = lse.regularization(A, y, N, lam)
print("LSE:")
cal_fittingline(res_lse)
cal_error(A, res_lse, y)
print()

# Steepest Descent
res_steep = std.descent(A, y, N, lam, lr=1e-4)
print("Steepest Descent Method: ")
cal_fittingline(res_steep)
cal_error(A, res_steep, y)
print()

# Newton's Method
res_newton = newton.optimize(A, y, N)
print("Newton's Method: ")
cal_fittingline(res_newton)
cal_error(A, res_newton, y)

# plotting
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].scatter(X, y, label= "data point", color='red')
axs[0].set_title("LSE")

axs[1].scatter(X, y, label= "data point", color='red')
axs[1].set_title("Steepest Descent Method")

axs[2].scatter(X, y, label= "data point", color='red')
axs[2].set_title("Newton Method")

x_range = np.linspace(min(X), max(X), 200)
A = design_matrix(x_range, N)
Ax1 = A @ res_lse
Ax2 = A @ res_steep
Ax3 = A @ res_newton
axs[0].plot(x_range, Ax1, label=f"{"LSE"} Fitting Line", color="black")
axs[1].plot(x_range, Ax2, label=f"{"Steep Descent Method"} Fitting Line", color="black")
axs[2].plot(x_range, Ax3, label=f"{"Newton Method"} Fitting Line", color="black")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.tight_layout()
plt.show()
