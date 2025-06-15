import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def project_onto_eigen(X, mean_face, top_eigenvectors):
    if mean_face is None:
        return np.dot(X, top_eigenvectors)
    return np.dot(X - mean_face, top_eigenvectors)

def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def rbf_kernel(X, Y, gamma=1e-3):
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * pairwise_sq_dists)

def poly_kernel(X, Y, degree=2, coef0=1):
    return (np.dot(X, Y.T) + coef0) ** degree

def center_train_kernel(K):
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    return K - np.dot(one_n, K) - np.dot(K, one_n) + np.dot(np.dot(one_n, K), one_n)

def center_test_kernel(K_test, K_train):
    N = K_train.shape[0]
    K_test_centered = K_test - np.mean(K_train, axis=0) - np.mean(K_test, axis=1, keepdims=True) + np.mean(K_train)
    return K_test_centered

def pca(X, n_components):
    mean_face = np.mean(X, axis=0, keepdims=True)  # 每一個 pixel 的平均值，長度 = D
    print("mean_face.shape:", mean_face.shape)  # 檢查平均臉的形狀
    X_centered = X - mean_face  # 中心化數據
    print("X_centered.shape:", X_centered.shape)  # 檢查中心化後的數據形狀

    # Kernel Trick for Linear PCA
    covariance_matrix = np.dot(X_centered, X_centered.T) / len(X[0])  # 計算協方差矩陣 S
    print("covariance_matrix.shape:", covariance_matrix.shape)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # 計算特徵值和特徵向量
    print("eigenvalues.shape:", eigenvalues.shape)

    # 將特徵值和特徵向量按特徵值從大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 投影回 high dimensional space ()
    proj_eigenvectors = np.dot(X_centered.T, eigenvectors)

    # 選擇前 n_components 個特徵向量
    top_eigenvectors = proj_eigenvectors[:, :n_components]
    print("top_eigenvectors.shape:", top_eigenvectors.shape)

    # Normalize eigenvectors
    top_eigenvectors /= np.linalg.norm(top_eigenvectors, axis=0, keepdims=True)
    return top_eigenvectors, mean_face

def pca_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, mean_face, k=1):
    X_train_proj = project_onto_eigen(X_train, mean_face, top_eigenvectors)
    X_test_proj = project_onto_eigen(X_test, mean_face, top_eigenvectors)

    correct_predictions = 0
    for i in range(X_test_proj.shape[0]):
        test = X_test_proj[i]
        distances = np.linalg.norm(X_train_proj - test, axis=1)

        # Get the k nearest neighbors
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = y_train[top_k_indices]

        # Majority voting for the predicted label
        counts = np.bincount(top_k_labels)
        predicted_label = np.argmax(counts)

        if predicted_label == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / X_test_proj.shape[0]
    print(f"PCA Accuracy (k={k}): {accuracy * 100:.2f}%")

def pca_reconstruction(X_train, top_eigenvectors, mean_face, image_shape, num_samples):
    # 投影到 PCA 空間
    X_test_proj = project_onto_eigen(X_train, mean_face, top_eigenvectors)
    # 投影回 high dim，再中心化
    X_test_recon = np.dot(X_test_proj, top_eigenvectors.T) + mean_face
    indices = np.random.choice(len(X_train), num_samples, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # 4 行 5 列

    for i in range(num_samples):
        col = i % 5  # 第 i 張圖對應的 column（0~4）
        
        # 原圖的行：第 0 行顯示前 5 張，後 5 張顯示在第 2 行
        row_original = 0 if i < 5 else 2
        axes[row_original, col].imshow(X_train[indices[i]].reshape(image_shape), cmap='gray')
        axes[row_original, col].set_title(f"Original {i+1}", fontsize=10)
        axes[row_original, col].axis('off')

        # 重建圖的行：第 1 行與第 3 行
        row_recon = 1 if i < 5 else 3
        axes[row_recon, col].imshow(X_test_recon[indices[i]].reshape(image_shape), cmap='gray')
        axes[row_recon, col].set_title(f"Reconstructed {i+1}", fontsize=10)
        axes[row_recon, col].axis('off')

    plt.tight_layout()
    plt.savefig("kernel_eigenface/recon/pca.png", dpi=300)
    plt.show()

def pca_kernel(X, kernel_func, n_components=25):
    # Compute kernel matrix and center it
    K = kernel_func(X, X)  # shape: (n_samples, n_samples)
    K_centered = center_train_kernel(K) 

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    sort_index = np.argsort(-eigenvalues)  # sort eigenvalues in descending order
    eigenvalues = eigenvalues[sort_index]
    eigenvectors = eigenvectors[:, sort_index]
    
    # 選擇前 n_components 個特徵值和向量
    top_eigenvalues = eigenvalues[:n_components]
    top_eigenvectors = eigenvectors[:, :n_components]

    # Normalize eigenvectors
    top_eigenvectors = top_eigenvectors / np.linalg.norm(top_eigenvectors, axis=0)
    return top_eigenvectors, K_centered

def pca_kernel_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, K_centered, kernel_func, k):
    K_train_raw = kernel_func(X_train, X_train)
    K_test_raw = kernel_func(X_test, X_train)
    
    K_train = center_train_kernel(K_train_raw)
    K_test = center_test_kernel(K_test_raw, K_train_raw)

    # Project training and testing data onto the top eigenvectors
    X_train_proj = np.dot(K_train, top_eigenvectors)
    X_test_proj = np.dot(K_test, top_eigenvectors)

    # knn classification
    correct_predictions = 0
    for i in range(X_test_proj.shape[0]):
        test = X_test_proj[i]
        distances = np.linalg.norm(X_train_proj - test, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = y_train[top_k_indices].astype(int)
        top_k_labels = y_train[top_k_indices]
        predicted_label = np.argmax(np.bincount(top_k_labels))
        if predicted_label == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / X_test_proj.shape[0]
    print(f"Kernel PCA Accuracy (k={k}): {accuracy * 100:.2f}%")


def lda(X, y, n_components):
    n_samples, n_features = X.shape
    print(n_samples, n_features)
    class_labels = np.unique(y)

    # Calculate the mean of each data point
    mean_total = np.mean(X, axis=0)

    # Scatter within class SW, scatter between class SB
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))

    for c in class_labels:
        Xc = X[y == c]
        mean_c = np.mean(Xc, axis=0)
        SW += (Xc - mean_c).T @ (Xc - mean_c)
        n_c = Xc.shape[0]
        mean_diff = (mean_c - mean_total).reshape(-1, 1)
        SB += n_c * (mean_diff @ mean_diff.T)

    # Solve generalized eigenvalue problem from SW⁻¹SB
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW) @ SB) # pseudo inverse for numerical stability
    idx = np.argsort(-eigvals.real)
    eigvecs = eigvecs[:, idx[:n_components]].real
    return eigvecs

def lda_recognition(X_train, y_train, X_test, y_test, mean_face, pca_components, lda_components, k=1):
    # Project to PCA
    X_pca_train = project_onto_eigen(X_train, mean_face, pca_components)
    X_pca_test = project_onto_eigen(X_test, mean_face, pca_components)

    # Project to LDA 
    X_lda_train = project_onto_eigen(X_pca_train, None, lda_components)
    X_lda_test = project_onto_eigen(X_pca_test, None, lda_components)

    # 計算測試樣本到所有訓練樣本的距離
    dists = cdist(X_lda_test, X_lda_train)  # shape: (n_test, n_train)

    # Get the k nearest neighbors
    knn_indices = np.argsort(dists, axis=1)[:, :k]
    knn_labels = y_train[knn_indices]

    # Predict labels
    if k == 1:
        y_pred = knn_labels.flatten()
    else:
        y_pred = []
        for row in knn_labels:
            counts = np.bincount(row)
            y_pred.append(np.argmax(counts))
        y_pred = np.array(y_pred)

    accuracy = np.mean(y_pred == y_test)
    print(f"[LDA + PCA] Face Recognition Accuracy (k={k}): {accuracy * 100:.2f}%")

def lda_reconstruction(X_train, mean_face, pca_components, lda_components, image_shape, num_samples):
    # Project to PCA
    X_pca = project_onto_eigen(X_train, mean_face, pca_components)
    # Project to LDA 
    X_lda = project_onto_eigen(X_pca, None, lda_components)
    # 投影回 high dim，再中心化
    X_pca_recon = X_lda @ lda_components.T  # (n_test, 100)
    X_recon = X_pca_recon @ pca_components.T + mean_face  # (n_test, 45045)

    indices = np.random.choice(len(X_train), num_samples, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # 4 行 5 列

    for i in range(num_samples):
        col = i % 5  # 第 i 張圖對應的 column（0~4）
        
        # 原圖的行：第 0 行顯示前 5 張，後 5 張顯示在第 2 行
        row_original = 0 if i < 5 else 2
        axes[row_original, col].imshow(X_train[indices[i]].reshape(image_shape), cmap='gray')
        axes[row_original, col].set_title(f"Original {i+1}", fontsize=10)
        axes[row_original, col].axis('off')

        # 重建圖的行：第 1 行與第 3 行
        row_recon = 1 if i < 5 else 3
        axes[row_recon, col].imshow(X_recon[indices[i]].reshape(image_shape), cmap='gray')
        axes[row_recon, col].set_title(f"Reconstructed {i+1}", fontsize=10)
        axes[row_recon, col].axis('off')

    plt.tight_layout()
    plt.savefig("kernel_eigenface/recon/lda.png", dpi=300)
    plt.show()

def lda_kernel(X, y, kernel_func, n_components=25):
    n_samples = X.shape[0]
    classes = np.unique(y)

    # Compute kernel matrix and center it
    K = kernel_func(X, X)
    K_centered = center_train_kernel(K)

    # Calculate the mean of each data point
    overall_mean = np.mean(K_centered, axis=0, keepdims=True)

    # Scatter within class SW, scatter between class SB
    S_B = np.zeros((n_samples, n_samples))
    S_W = np.zeros((n_samples, n_samples))

    for c in classes:
        idx = np.where(y == c)[0]
        K_c = K_centered[idx]
        n_c = len(idx)
        mean_c = np.mean(K_c, axis=0, keepdims=True)

        mean_diff = mean_c - overall_mean
        S_B += n_c * (mean_diff.T @ mean_diff)

        for i in range(n_c):
            diff = (K_c[i:i+1] - mean_c)
            S_W += diff.T @ diff

    # Generalized eigenvalue problem with pseudo-inverse
    S_W_pinv = np.linalg.pinv(S_W)
    matrix = S_W_pinv @ S_B

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)
    top_indices = np.argsort(eigvals)[-n_components:]
    top_eigenvectors = eigvecs[:, top_indices]

    return top_eigenvectors, K_centered, K

def lda_kernel_recognition(K_centered, X_train, y_train, X_test, y_test, top_eigenvectors, kernel_func, K_train_raw=None, k=1, visualize=True):
    # Compute kernel and center it
    if K_train_raw is None:
        K_train_raw = kernel_func(X_train, X_train)
    K_test_raw = kernel_func(X_test, X_train)
    K_test = center_test_kernel(K_test_raw, K_train_raw)

    # Project training and testing data onto the top eigenvectors
    X_train_proj = np.dot(K_centered, top_eigenvectors)
    X_test_proj = np.dot(K_test, top_eigenvectors)

    # Use k-NN for classification and calculate accuracy
    correct_predictions = 0
    for i in range(X_test_proj.shape[0]):
        test = X_test_proj[i]
        distances = np.linalg.norm(X_train_proj - test, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = y_train[top_k_indices].astype(int)
        predicted_label = np.argmax(np.bincount(top_k_labels))
        if predicted_label == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / X_test_proj.shape[0]
    print(f"Kernel LDA Accuracy (k={k}): {accuracy * 100:.2f}%")