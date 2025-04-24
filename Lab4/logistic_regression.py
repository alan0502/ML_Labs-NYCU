import numpy as np
import matplotlib.pyplot as plt
import optimization as op
import pandas as pd

n = int(input("Enter the number of data points: "))
mx1, vx1 = 1, 2#map(int, input("mx1, vx1: ").split())
my1, vy1 = 1, 2#map(int, input("my1, vy1: ").split())
mx2, vx2 = 10, 2#map(int, input("mx2, vx2: ").split())
my2, vy2 = 10, 2#map(int, input("my2, vy2: ").split())

def multivariate_gaussian_generator(mux, varx, muy, vary):
    sigmax = np.sqrt(varx)
    sigmay = np.sqrt(vary)
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    # Box-Muller transform
    Z0 = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    Z1 = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V)
    # Transform to the desired mean and variance
    X = mux + sigmax * Z0
    Y = muy + sigmay * Z1
    return X, Y

def confusion_matrix(y_true, p_class, w):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(p_class)):
        if p_class[i] == 1 and y_true[i] == 1:
            true_positive += 1
        elif p_class[i] == 0 and y_true[i] == 0:
            true_negative += 1
        elif p_class[i] == 1 and y_true[i] == 0:
            false_positive += 1
        elif p_class[i] == 0 and y_true[i] == 1:
            false_negative += 1
    data = {
        "Predict cluster 1": [true_positive, false_negative],
        "Predict cluster 2": [false_positive, true_negative]
    }
    index = ["Is cluster 1", "Is cluster 2"]
    df = pd.DataFrame(data, index=index)

    print(f'w: {w.flatten()}')
    #print(f'True Positive: {true_positive}')
    #print(f'True Negative: {true_negative}')
    #print(f'False Positive: {false_positive}')
    #print(f'False Negative: {false_negative}')
    print("Confusion Matrix:")
    print(df)
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    print(f'Sensitivity (Successfully predict cluster 1): {sensitivity:.5f}')
    print(f'Specificity (Successfully predict cluster 2): {specificity:.5f}')
    return true_positive, true_negative, false_positive, false_negative, sensitivity, specificity

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

D1 = []
D2 = []
#np.random.seed(0)
for i in range(n):
    x1, y1 = multivariate_gaussian_generator(mx1, vx1, my1, vy1)
    D1.append([x1, y1])
    x2, y2 = multivariate_gaussian_generator(mx2, vx2, my2, vy2)
    D2.append([x2, y2])
D1 = np.array(D1)
D2 = np.array(D2)
X = np.vstack([
    np.hstack([np.ones((n, 1)), D1]),
    np.hstack([np.ones((n, 1)), D2])
])
y = np.array([0]*n + [1]*n).reshape(-1, 1) #[0, 1, 0, 1, ...] shape: (2n, 1)
#print(y[:5])  # Print first 5 labels
#print(y.shape)  # shape: (2n, 1)
w = np.random.randn(3, 1) * 0.01
#print(w.shape)  # shape: (3, 1)
eta = 0.01  # Learning rate 
epoch = 10000  # Number of epochs for gradient descent
w_grad, p_grad = op.gradient_descent(w, X, y, eta, epoch)
w_newton, p_newton = op.newtons_method(w, X, y, eta, epoch)

pred_grad = np.where(p_grad >= 0.5, 1, 0)  # shape: (2n, 1)
pred_newton = np.where(p_newton >= 0.5, 1, 0)  # shape: (2n, 1)

print()
print("Gradient Descent:")
grad_tp, grad_tn, grad_fp, grad_fn, grad_sensitivity, grad_specificity = confusion_matrix(y, pred_grad, w_grad)
print("----------------------------------------------------------------------")
print("Newton's Method:")
newton_tp, newton_tn, newton_fp, newton_fn, newton_sensitivity, newton_specificity = confusion_matrix(y, pred_newton, w_newton)

plt.figure(figsize=(12, 5))  
# Ground Truth
plt.subplot(1, 3, 1)
plt.scatter(D1[:, 0], D1[:, 1], color='red', label='D1 (class 0)', alpha=0.6)
plt.scatter(D2[:, 0], D2[:, 1], color='blue', label='D2 (class 1)', alpha=0.6)
plt.title("Ground Truth")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')

# Predicted Classification using Gradient Descent
plt.subplot(1, 3, 2)
# 把 X 中的點根據預測類別分開（記得 X 有加 bias，要畫的是 X[:,1], X[:,2]）
plt.scatter(X[pred_grad.flatten() == 0][:, 1], X[pred_grad.flatten() == 0][:, 2], color='red', label='D1 (class 0)', alpha=0.6)
plt.scatter(X[pred_grad.flatten() == 1][:, 1], X[pred_grad.flatten() == 1][:, 2], color='blue', label='D2 (class 1)', alpha=0.6)
plt.title("Gradient Descent Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')

# Predicted Classification using Newton's Method
plt.subplot(1, 3, 3)
# 把 X 中的點根據預測類別分開（記得 X 有加 bias，要畫的是 X[:,1], X[:,2]）
plt.scatter(X[pred_newton.flatten() == 0][:, 1], X[pred_newton.flatten() == 0][:, 2], color='red', label='D1 (class 0)', alpha=0.6)
plt.scatter(X[pred_newton.flatten() == 1][:, 1], X[pred_newton.flatten() == 1][:, 2], color='blue', label='D2 (class 1)', alpha=0.6)
plt.title("Newton's Method Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
#plt.savefig("result_plot/case2.png")
plt.tight_layout()
plt.show()

    
