import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from libsvm.svmutil import *

def cal_linear_rbf_kernel(X1, X2, gamma):
    linear = np.dot(X1, X2.T)
    rbf = -2 * np.dot(X1, X2.T) + np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)
    rbf = np.exp(-gamma * rbf)
    return linear + rbf

def linear(X_train, y_train, X_test, y_test, C_list):
    final_C = 0
    best_acc = -1
    acc_matrix = np.zeros((len(C_list), 1))
    for i, C in enumerate(C_list):
        print(f"Training SVM with linear kernel, C={C}...")
        start = time.time()
        acc = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 0 -c {C} -v 5 -q')
        end = time.time()
        acc_matrix[i, 0] = acc
        print(f"CV Accuracy = {acc:.2f}%, Time: {end - start:.2f} sec")

        if acc > best_acc:
            best_acc = acc
            final_C = C
            print("Update best parameters\n")

    print(f"Best parameters: C={final_C}, CV acc={best_acc:.2f}%")
    # Final model training
    model = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 0 -c {final_C}')
    _, train_acc, _ = svm_predict(y_train.tolist(), X_train.tolist(), model)
    _, test_acc, _ = svm_predict(y_test.tolist(), X_test.tolist(), model)

    print(f"Train acc = {train_acc[0]:.2f}%, Test acc = {test_acc[0]:.2f}%")
    #svm_save_model('result/SVM/task2_linear.txt', model)

    return acc_matrix

def poly(X_test, y_test, X_train, y_train, C_list, gamma_list, d_list):
    final_C = final_gamma = final_d = 0
    best_acc = -1
    acc_matrix = np.zeros((len(C_list), len(gamma_list), len(d_list)))
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            for k, d in enumerate(d_list):
                print(f"Training SVM with polynomial kernel, C={C}, gemma={gamma}, d={d}...")
                start = time.time()
                acc = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 1 -c {C} -d {d} -g {gamma} -v 5 -q')
                end = time.time()
                acc_matrix[i, j, k] = acc
                print(f"CV Accuracy = {acc:.2f}%, Time: {end - start:.2f} sec")
                
                if acc > best_acc:
                    best_acc = acc
                    final_C = C
                    final_d = d
                    final_gamma = gamma
                    print("Update best parameters\n")

    print(f"Best parameters: C={final_C}, gamma={final_gamma}, d={final_d}, CV acc={best_acc:.2f}%")
    # Final model training
    model = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 1 -c {final_C} -g {gamma} -d {final_d}')
    _, train_acc, _ = svm_predict(y_train.tolist(), X_train.tolist(), model)
    _, test_acc, _ = svm_predict(y_test.tolist(), X_test.tolist(), model)

    print(f"Train acc = {train_acc[0]:.2f}%, Test acc = {test_acc[0]:.2f}%")
    #svm_save_model('result/SVM/task2_poly.txt', model)
    return acc_matrix

def rbf(X_test, y_test, X_train, y_train, C_list, gamma_list):
    final_C = final_gamma = 0
    best_acc = -1
    acc_matrix = np.zeros((len(C_list), len(gamma_list)))
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            print(f"Training SVM with RBF kernel, C={C}, gamma={gamma}...")
            start = time.time()
            acc = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 2 -c {C} -g {gamma} -v 5 -q')
            end = time.time()
            acc_matrix[i, j] = acc
            print(f"CV Accuracy = {acc:.2f}%, Time: {end - start:.2f} sec")

            if acc > best_acc:
                best_acc = acc
                final_C = C
                final_gamma = gamma
                print("Update best parameters\n")

    print(f"Best parameters: C={final_C}, gamma={final_gamma}, CV acc={best_acc:.2f}%")
    # Final model training
    model = svm_train(y_train.tolist(), X_train.tolist(), f'-s 0 -t 2 -c {final_C} -g {final_gamma}')
    _, train_acc, _ = svm_predict(y_train.tolist(), X_train.tolist(), model)
    _, test_acc, _ = svm_predict(y_test.tolist(), X_test.tolist(), model)

    print(f"Train acc = {train_acc[0]:.2f}%, Test acc = {test_acc[0]:.2f}%")
    #svm_save_model('result/SVM/task2_rbf.txt', model)

    plt.figure(figsize=(8, 6))
    sns.heatmap(acc_matrix, annot=True, fmt=".2f", xticklabels=gamma_list, yticklabels=C_list, cmap="YlGnBu")
    plt.title("5-fold CV Accuracy (%) for RBF Kernel")
    plt.xlabel("Gamma")
    plt.ylabel("C")
    #plt.savefig('result/SVM/task2_rbf.png')
    plt.show()
    return acc_matrix