import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

def kmeans_plusplus_init_kernel(kernel, k):
    N = kernel.shape[0]
    centers = []
    # Random select the center
    first = np.random.randint(0, N)
    centers.append(first)
    
    for _ in range(1, k):
        dists = []
        for i in range(N):
            min_dist = float('inf')
            for c in centers:
                # kernel distance^2 = k(x,x) - 2k(x,c) + k(c,c)
                dist_sq = kernel[i, i] - 2 * kernel[i, c] + kernel[c, c]
                min_dist = min(min_dist, dist_sq)
            dists.append(min_dist)
        dists = np.array(dists)
        probs = dists / np.sum(dists)
        new_center = np.random.choice(N, p=probs)
        centers.append(new_center)
    
    # 依據距離最小 assign labels
    dists_to_centers = np.zeros((N, k))
    for idx, c in enumerate(centers):
        dists_to_centers[:, idx] = kernel.diagonal() - 2 * kernel[:, c] + kernel[c, c]
    labels = np.argmin(dists_to_centers, axis=1)
    return labels


def kmeans_plusplus_init_spectral(features, k, retries=10):
    N = features.shape[0]
    best_labels = None
    best_loss = float('inf')

    for attempt in range(retries):
        centers = []

        # Random select the center
        first_center_idx = np.random.randint(0, N)
        centers.append(features[first_center_idx])

        # 根據距離平方機率分布選新中心，距離dist越遠被選成center越高
        for _ in range(1, k):
            dist = []
            # 對每個 datapoint 算離他最近距離的中心的距離
            for x in features:
                min_dist_sq = float('inf')
                for c in centers:
                    d = np.linalg.norm(x - c)**2
                    if d < min_dist_sq:
                        min_dist_sq = d
                dist.append(min_dist_sq)

            dist = np.array(dist)
            probs = dist / np.sum(dist)
            new_center_idx = np.random.choice(N, p=probs)
            centers.append(features[new_center_idx])

        # Assign labels
        centers = np.array(centers)
        dists_to_centers = np.linalg.norm(features[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists_to_centers, axis=1)

        # 檢查群大小是否合理，若其中一群只有一點，無效
        cluster_sizes = np.bincount(labels, minlength=k)
        if np.min(cluster_sizes) <= 1:
            print(f"[Attempt {attempt+1}] Discarded: one cluster only has 1 point.")
            continue

        # 計算 loss：每點到其中心的距離平方總和
        loss = np.sum(np.linalg.norm(features - centers[labels], axis=1) ** 2)
        print(f"[Attempt {attempt+1}] OK, loss = {loss:.2f}, min cluster size = {np.min(cluster_sizes)}")

        if loss < best_loss:
            best_loss = loss
            best_labels = labels

    if best_labels is None:
        print("[Warning] No good initialization found. Fallback to random init.")
        best_labels = np.random.randint(0, k, size=N)

    # 可視化初始化結果
    plt.imshow(best_labels.reshape(100, 100), cmap='tab10')
    plt.title("Initial Labels from Best K-means++")
    plt.show()

    return best_labels

def plot_eigen_projection(H, labels, dir_name):
    plt.figure(figsize=(6, 5))
    plt.scatter(H[:, 0], H[:, 1], c=labels, cmap='tab10', s=5)
    #plt.title("title")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.grid(True)
    plt.savefig(f"{dir_name}/eigenspace_feature.png")
    plt.show()

def plot_single_eigenvector(H, labels, eig_idx, dir_name):
    plt.figure(figsize=(8, 2))
    plt.scatter(range(len(H)), H[:, eig_idx], c=labels, cmap='tab10', s=10)
    plt.title(f"Eigenvector {eig_idx+1} (sorted by index)")
    plt.xlabel("Data Index")
    plt.ylabel(f"Value in Eigenvector {eig_idx+1}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/eigenspace{eig_idx+1}.png")
    plt.show()

def make_gif_from_frames(frame_dir, output_gif, duration=500):
    frame_paths = sorted(glob.glob(f"{frame_dir}/epoch_*.png"))
    frame_paths.sort(key=os.path.getmtime)
    frames = [Image.open(fp).convert("RGB") for fp in frame_paths]

    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"GIF saved to: {output_gif}")
    else:
        print(f"No PNG frames found in {frame_dir}")

def save_epoch_visual(labels, image_size, epoch, save_dir="frames"):
    os.makedirs(save_dir, exist_ok=True)
    segmented = labels.reshape(image_size)
    plt.imshow(segmented, cmap='tab10')
    plt.title(f"Epoch {epoch+1}")
    plt.axis('off')
    plt.savefig(f"{save_dir}/epoch_{epoch+1:03d}.png")
    plt.close()

def kernel_kmeans(epoch, k, kernel, image_size, dir_name, kmeanspp):
    N = kernel.shape[0]
    if kmeanspp == 0:
        labels = np.random.randint(0, k, size=N)
    else:
        labels = kmeans_plusplus_init_kernel(kernel, k)
    losses = []
    for e in range(epoch):
        print(f'Epoch {e+1}: ')
        label_idx = [np.where(labels == c)[0] for c in range(k)]
        #print(label_idx)
        #print(len(label_idx)) # len = k
        term2_idx = []
        term3_c = []
        for idx_c in label_idx:
            if len(idx_c) == 0:
                term2_idx.append(np.zeros(N))  # 避免 term2 出錯
                term3_c.append(0.0)
                continue
            term2_idx.append(np.sum(kernel[:, idx_c], axis=1))  # 所有點與這群的 sum（for term2）
            term3_c.append(np.sum(kernel[np.ix_(idx_c, idx_c)]))  # term3
        new_labels = []
        total_loss = 0
        for i in range(100):
            for j in range(100):
                idx = i*100 + j
                #print("(", i, ", ", j, ")")
                #print(idx)
                dist = []
                # first term
                term1 = kernel[idx, idx]
                #print(first)
                for c in range(k):
                    # idx_c: cluster c 的 datapoint index
                    idx_c = label_idx[c]
                    # second term
                    #term2 = 0
                    #for id in idx_c:
                    #    term2 += kernel[id, idx]
                    #term2 = np.sum(kernel[idx_c, idx])
                    term2 = term2_idx[c][idx]

                    # third term
                    #term3 = 0
                    #for pid in idx_c:
                    #    for qid in idx_c:
                    #        term3 += kernel[pid, qid]
                    #term3 = np.sum(kernel[np.ix_(idx_c, idx_c)])
                    term3 = term3_c[c]

                    if len(idx_c) > 0:
                        final_term = term1 - 2*(1/len(idx_c))*term2 + (1/(len(idx_c))**2)*term3
                    else:
                        final_term = float('inf')  # 避免空群錯誤
                    dist.append(final_term)
                new_labels.append(np.argmin(dist))
                total_loss += dist[np.argmin(dist)]
                #print(dist)
                #print(np.argmin(dist))
        #print(f'Original Labels: {labels}')
        #print(f'New Labels: {new_labels}')
        losses.append(total_loss)
        print(f'Total kernel K-means loss: {total_loss:.2f}')
        save_epoch_visual(np.array(new_labels), image_size, e, dir_name)
        if np.array_equal(new_labels, labels):
            print(f"Converged at epoch {e}")
            break
        labels = np.array(new_labels)
    return labels, losses

def spectral_ratio_cut(W, epoch, k, image_size, dir_name, kmeanspp):
    os.makedirs(dir_name, exist_ok=True)
    N = W.shape[0]
    #print(N)
    # Step 1: Degree and Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Step 2: Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)  # Since L is symmetric (use eigh)
    print(eigvals[:10])
    #print(eigvecs)
    H = eigvecs[:, 1:k+1]  # take 2nd to Nth eigenvectors
    #print(H)
    #print(H.shape)
    
    # Step 3: Initialize cluster labels
    if kmeanspp == 0:
        labels = np.random.randint(0, k, size=N)
    else:
        labels = kmeans_plusplus_init_spectral(H, k)
    losses = []
    #print(H[labels == 0])
    for e in range(epoch):
        print(f"Epoch {e+1}:")
        for c in range(k):
            print(f"Cluster {c}: {np.sum(labels == c)} points")
        centroids = []
        for c in range(k):
            if np.any(labels == c):
                centroids.append(H[labels == c].mean(axis=0))
            else:
                centroids.append(np.zeros(H.shape[1]))
        centroids = np.array(centroids)

        new_labels = np.argmin(((H[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2), axis=1)
        total_loss = np.sum(np.linalg.norm(H - centroids[new_labels], axis=1) ** 2)
        losses.append(total_loss)
        print(f"Spectral clustering loss: {total_loss:.2f}")
        save_epoch_visual(np.array(new_labels), image_size, e, dir_name)

        if np.array_equal(new_labels, labels):
            print(f"Converged at epoch {e+1}")
            break

        labels = new_labels.copy()
    plot_eigen_projection(H, labels, dir_name)
    for i in range(k):
        plot_single_eigenvector(H, labels, i, dir_name)
    return labels, losses

def spectral_normalized_cut(W, epoch, k, image_size, dir_name, kmeanspp):
    os.makedirs(dir_name, exist_ok=True)
    N = W.shape[0]
    # Step 1: Degree matrix and normalized Laplacian
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-8))  # avoid division by zero
    L = D - W
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # symmetric normalized Laplacian

    # Step 2: Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    H = eigvecs[:, 1:k+1]  # skip trivial all-1 eigenvector

    # Step 3: Row-normalize H (optional but standard for Ncut)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    # Step 4: Spectral K-means (same as before)
    if kmeanspp == 0:
        labels = np.random.randint(0, k, size=N)
    else:
        labels = kmeans_plusplus_init_spectral(H, k)
    losses = []

    for e in range(epoch):
        print(f"Epoch {e+1}:")
        for c in range(k):
            print(f"Cluster {c}: {np.sum(labels == c)} points")
        centroids = []
        for c in range(k):
            if np.any(labels == c):
                centroids.append(H[labels == c].mean(axis=0))
            else:
                centroids.append(np.zeros(H.shape[1]))
        centroids = np.array(centroids)

        new_labels = np.argmin(((H[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2), axis=1)
        total_loss = np.sum(np.linalg.norm(H - centroids[new_labels], axis=1) ** 2)
        losses.append(total_loss)
        print(f"Normalized Cut clustering loss: {total_loss:.2f}")

        save_epoch_visual(np.array(new_labels), image_size, e, dir_name)

        if np.array_equal(new_labels, labels):
            print(f"Converged at epoch {e+1}")
            break

        labels = new_labels.copy()
    plot_eigen_projection(H, labels, dir_name)
    for i in range(k):
        plot_single_eigenvector(H, labels, i, dir_name)
    return labels, losses