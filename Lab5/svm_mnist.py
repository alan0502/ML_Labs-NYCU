import numpy as np
import pandas as pd
from libsvm.svmutil import *
import time
import matplotlib.pyplot as plt
import seaborn as sns
from grid_search import *

def linear_rbf_kernel(X1, X2, gamma):
    linear = np.dot(X1, X2.T)
    rbf = -2 * np.dot(X1, X2.T) + np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)
    rbf = np.exp(-gamma * rbf)
    return linear + rbf

# Load the dataset
X_train = pd.read_csv('data/X_train.csv', header=None).values
y_train = pd.read_csv('data/Y_train.csv', header=None).values.flatten()
X_test = pd.read_csv('data/X_test.csv', header=None).values
y_test = pd.read_csv('data/Y_test.csv', header=None).values.flatten()

# Normalize the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)
#print(X_train.shape, y_train.shape)
#print(X_train[:5], y_train[:5])
#print(X_test.shape, y_test.shape)
#print(X_test[0])

# Task 1
kernel_types = ['linear', 'polynomial', 'rbf']
kernel_params = ['-s 0 -t 0 -g 0.001', '-s 0 -t 1 -g 0.001', '-s 0 -t 2 -g 0.001']

for i, kernel in enumerate(kernel_types):
    print(f"Training SVM with {kernel} kernel...")
    start = time.time()
    model = svm_train(y_train.tolist(), X_train.tolist(), kernel_params[i])
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
    print("Training:")
    _, train_acc, _ = svm_predict(y_train.tolist(), X_train.tolist(), model)
    print("Testing:")
    _, test_acc, _ = svm_predict(y_test.tolist(), X_test.tolist(), model)
    print()

# Task 2
C_list = [0.1, 1, 10]
gamma_list = [0.001, 0.01, 0.1]
d_list = [0, 1, 2]
linear_acc = linear(X_train, y_train, X_test, y_test, C_list)
poly_acc = poly(X_train, y_train, X_test, y_test, C_list, gamma_list, d_list)
rbf_acc = rbf(X_train, y_train, X_test, y_test, C_list, gamma_list)

# Task 3
final_C = final_gamma = 0
best_acc = -1
acc_matrix = np.zeros((len(C_list), len(gamma_list)))
for i, gamma in enumerate(gamma_list):
    K_train = linear_rbf_kernel(X_train, X_train, gamma)
    K_train = np.hstack([np.arange(1, K_train.shape[0]+1).reshape(-1, 1), K_train])

    K_test = linear_rbf_kernel(X_test, X_train, gamma)
    K_test = np.hstack([np.arange(1, K_test.shape[0]+1).reshape(-1, 1), K_test])
    for j, C in enumerate(C_list):
        print(f"Training SVM with linear + RBF kernel, C={C}, gamma={gamma}...")
        start = time.time()
        acc = svm_train(y_train.tolist(), K_train.tolist(), f'-s 0 -t 4 -c {C} -g {gamma} -v 5 -q')
        acc_matrix[j, i] = acc
        end = time.time()
        print(f"Training time: {end - start:.2f} seconds")
        print(f"CV Accuracy = {acc:.2f}%, Time: {end - start:.2f} sec")
        if acc > best_acc:
            best_acc = acc
            final_C = C
            final_gamma = gamma
            print("Update best parameters\n")
print(f"Best parameters: C={final_C}, gamma={final_gamma}, CV acc={best_acc:.2f}%")
# Final model training
model = svm_train(y_train.tolist(), K_train.tolist(), f'-s 0 -t 4 -c {final_C} -g {final_gamma}')
        
print("Training:")
_, acc_train, _ = svm_predict(y_train.tolist(), K_train.tolist(), model)
print("Testing:")
_, acc_test, _ = svm_predict(y_test.tolist(), K_test.tolist(), model)

plt.figure(figsize=(8, 6))
sns.heatmap(acc_matrix, annot=True, fmt=".2f", xticklabels=gamma_list, yticklabels=C_list, cmap="YlGnBu")
plt.title("5-fold CV Accuracy (%) for RBF Kernel")
plt.xlabel("Gamma")
plt.ylabel("C")
#plt.savefig('result/SVM/linear_rbf.png')
plt.show()

#print(linear_acc)
#for i in range(poly_acc.shape[0]):
#    print(f"=== Slice {i} ===")
#    print(poly_acc[i])
