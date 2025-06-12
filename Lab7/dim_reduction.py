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

def poly_kernel(X, Y, degree=3, coef0=1):
    return (np.dot(X, Y.T) + coef0) ** degree

def center_train_kernel(K):
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    #return K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K - np.dot(one_n, K) - np.dot(K, one_n) + np.dot(np.dot(one_n, K), one_n)


def center_test_kernel(K_test_raw, K_train_raw):
    N = K_train_raw.shape[0]
    one_N = np.ones((N, N)) / N
    K_test_centered = K_test_raw - np.mean(K_train_raw, axis=0) - np.mean(K_test_raw, axis=1, keepdims=True) + np.mean(K_train_raw)
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

        # 取得前 k 個最近鄰
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = y_train[top_k_indices]

        # 多數投票決定預測結果
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
    n_samples = X.shape[0]
    print(X)
    K = kernel_func(X, X)  # shape: (n_samples, n_samples)
    print("K:")
    print(K)
    K_centered = center_train_kernel(K)  # 中心化核矩陣
    print("asa")
    print(K_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)  # 計算特徵值和特徵向量

    #print(f"Eigenvalues before sorting: {eigenvalues}")
    sort_index = np.argsort(-eigenvalues)  # sort eigenvalues in descending order
    eigenvalues = eigenvalues[sort_index]
    print(f"Eigenvalues after sorting: {eigenvalues}")
    eigenvectors = eigenvectors[:, sort_index]
    print(eigenvectors[:5, :5])  # 顯示前 5 個特徵向量
    
    # 選擇前 n_components 個特徵值和向量
    top_eigenvectors = eigenvectors[:, :n_components]
    top_eigenvalues = eigenvalues[:n_components]
    #print("iUiuiu")
    #print(top_eigenvalues)

    # Normalize eigenvectors
    #top_eigenvectors = top_eigenvectors / np.sqrt(top_eigenvalues[np.newaxis, :] + 1e-8)
    #top_eigenvectors = top_eigenvectors / np.linalg.norm(top_eigenvectors, axis=0)
    top_eigenvectors = top_eigenvectors / np.linalg.norm(top_eigenvectors, axis=0)

    #print("top_eigenvectors.shape:", top_eigenvectors.shape)

    orthogonality = np.dot(top_eigenvectors.T, top_eigenvectors)
    #print("Should be close to identity:\n", orthogonality)
    print("asaa")
    print(top_eigenvectors)
    return top_eigenvectors, K_centered

def pca_kernel_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, K_centered, kernel_func, k):
    K_train_raw = kernel_func(X_train, X_train)
    K_test_raw = kernel_func(X_test, X_train)
    #print(X_train.shape, X_test.shape)
    #print("K_train_raw shape:", K_train_raw.shape)
    #print("K_test_raw shape:", K_test_raw.shape)
    
    K_train = center_train_kernel(K_train_raw)
    K_test = center_test_kernel(K_test_raw, K_train_raw)

    #print("mean of K_train before center:", np.mean(K_train_raw))
    #print("mean of K_train after center:", np.mean(K_train))
    #print("mean of K_test before center:", np.mean(K_test_raw))
    #print("mean of K_test after center:", np.mean(K_test))

    # 投影到 PCA 空間
    X_train_proj = np.dot(K_train, top_eigenvectors)
    X_test_proj = np.dot(K_test, top_eigenvectors)
    #print("X_train_proj shape:", X_train_proj.shape)
    #print("X_test_proj shape:", X_test_proj.shape)

    plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], c=y_train, cmap='tab10', s=10)
    plt.title("Kernel PCA Projection (Train)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.show()

    # k-NN 多數投票分類
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
    class_labels = np.unique(y)
    n_classes = len(class_labels)

    # 計算整體平均
    mean_total = np.mean(X, axis=0)

    # 類內散佈矩陣 SW, 類間散佈矩陣 SB
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))

    for c in class_labels:
        Xc = X[y == c]
        mean_c = np.mean(Xc, axis=0)
        SW += (Xc - mean_c).T @ (Xc - mean_c)  # 類內散佈
        n_c = Xc.shape[0]
        mean_diff = (mean_c - mean_total).reshape(-1, 1)
        SB += n_c * (mean_diff @ mean_diff.T)  # 類間散佈

    # 求解廣義特徵值問題 SW⁻¹SB
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW) @ SB)
    idx = np.argsort(-eigvals.real)
    eigvecs = eigvecs[:, idx[:n_components]].real
    return eigvecs

def lda_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, mean_X, fisherfaces, k=1):
    # 將 train/test 投影進 Fisherface 空間
    X_train_lda = (X_train - mean_X) @ fisherfaces.T  # shape: (n_train, 25)
    X_test_lda = (X_test - mean_X) @ fisherfaces.T    # shape: (n_test, 25)
    # 計算測試樣本到所有訓練樣本的距離
    dists = cdist(X_test_lda, X_train_lda)  # shape: (n_test, n_train)

    # 找最近的 k 個鄰居
    knn_indices = np.argsort(dists, axis=1)[:, :k]
    knn_labels = y_train[knn_indices]

    # 預測：k=1 時直接取最近鄰；k>1 時取眾數
    if k == 1:
        y_pred = knn_labels.flatten()
    else:
        from scipy.stats import mode
        y_pred = mode(knn_labels, axis=1).mode.flatten()

    # 計算準確率
    accuracy = np.mean(y_pred == y_test)
    print(f"[LDA + PCA] Face Recognition Accuracy (k={k}): {accuracy * 100:.2f}%")

def lda_reconstruction(X_train, mean_face, pca_components, lda_components, image_shape, num_samples):
    # 投影到 PCA 空間
    X_pca = project_onto_eigen(X_train, mean_face, pca_components)
    # 投影到 LDA 空間
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

def lda_kernel(X, y, kernel_func, n_components=2):
    n_samples = X.shape[0]
    classes = np.unique(y)

    # Step 1: 計算中心化核矩陣
    K = kernel_func(X, X)
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Step 2: 類間與類內散佈
    overall_mean = np.mean(K_centered, axis=0, keepdims=True)
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

    # Step 3: 廣義特徵值問題轉標準特徵值問題
    reg = 1e-6 * np.eye(n_samples)
    S_W_inv = np.linalg.inv(S_W + reg)
    matrix = S_W_inv @ S_B

    # Step 4: 求 eigenvectors
    eigvals, eigvecs = np.linalg.eigh(matrix)
    top_indices = np.argsort(eigvals)[-n_components:]
    top_eigenvectors = eigvecs[:, top_indices]

    return top_eigenvectors, K_centered, K


def lda_kernel_recognition(
    K_centered,            # 已中心化的 K_train
    X_train, y_train,
    X_test, y_test,
    top_eigenvectors,
    kernel_func,
    K_train_raw=None,      # 可以提前給，否則內部自算
    k=1,
    visualize=True
):
    # Step 1: 投影 train
    X_train_proj = np.dot(K_centered, top_eigenvectors)

    # Step 2: 計算 K_test
    if K_train_raw is None:
        K_train_raw = kernel_func(X_train, X_train)
    K_test_raw = kernel_func(X_test, X_train)
    K_test = center_test_kernel(K_test_raw, K_train_raw)

    # Step 3: 投影 test
    X_test_proj = np.dot(K_test, top_eigenvectors)

    # Step 4: 視覺化 train 資料在前兩主成分
    if visualize:
        plt.figure(figsize=(6, 5))
        plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], c=y_train, cmap='tab10', s=10)
        plt.title("Kernel PCA Projection (Train)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Step 5: k-NN 分類
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