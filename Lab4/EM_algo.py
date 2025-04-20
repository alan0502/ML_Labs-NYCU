import struct
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import pandas as pd

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))  # 解析檔頭
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)  # 讀取數據並轉為 3D 陣列
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))  # 解析檔頭
        labels = np.frombuffer(f.read(), dtype=np.uint8)  # 讀取數據
    return labels

def confusion_matrix(y_true, p_class):
    print(y_true.shape)
    print(p_class.shape)
    for i in range(10):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for j in range(len(y_true)):
            if y_true[j] == i:
                if p_class[j] == i:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if p_class[j] == i:
                    false_positive += 1
                else:
                    true_negative += 1
        data = {
            f"Predict number {i}": [true_positive, false_negative],
            f"Predict not number {i}": [false_positive, true_negative]
        }
        index = [f"Is number {i}", f"Isn't number {i}"]
        df = pd.DataFrame(data, index=index)

        #print(f'True Positive: {true_positive}')
        #print(f'True Negative: {true_negative}')
        #print(f'False Positive: {false_positive}')
        #print(f'False Negative: {false_negative}')
        print()
        print(f"Confusion Matrix {i}:")
        print(df)
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        print(f'Sensitivity (Successfully predict number {i}): {sensitivity:.5f}')
        print(f'Specificity (Successfully predict not number {i}): {specificity:.5f}')
        print()
        print("----------------------------------------------------------------------")


def print_digit_imagination_from_pi(pi, digit_to_cluster, threshold=0.5):
    for digit in range(10):
        if digit not in digit_to_cluster:
            print(f"\n🚫 Digit {digit} 沒有被指派到任何 cluster，跳過。")
            continue

        cluster = digit_to_cluster[digit]
        prob_map = pi[cluster]
        binary_map = (prob_map > threshold).astype(int).reshape(28, 28)

        print(f"\n🤖 Model's Imagination of Digit {digit} (from Cluster {cluster}):")
        for row in binary_map:
            print(' '.join(str(x) for x in row))


# Read MNIST data
train_images = load_mnist_images("inputimage/train-images.idx3-ubyte_")
train_labels = load_mnist_labels("inputimage/train-labels.idx1-ubyte_")
bin_images = np.where(train_images > 127 , 1, 0)  # Convert to binary images
bin_images = bin_images.reshape(60000, 28*28)
#print(bin_images.shape)  # Shape: (60000, 784)
#print(bin_images[4])
#print(train_labels[4])
#print(train_labels.shape)
epochs = 0
num_groups = 10
num_images = 60000
num_pixels = 28 * 28
lam = np.ones(10) * 0.1
pi = np.random.rand(num_groups, num_pixels) * 0.5 + 0.25  # 讓初始值在 [0.25, 0.75]
wi = np.zeros((60000, num_groups))
threshold = 10
while True:
    print("========================================")
    epochs += 1
    print(f"Epoch {epochs}:")
    prev_pi = pi.copy()
    # E-step
    for i in range(num_images):
        if(i+1) % 1000 == 0:
            # 每 1000 張圖片印一次進度
            print(f"Processing image {i+1}/{num_images}...")
        log_lam = np.zeros(num_groups)  # 改成 log-likelihood 累加

        for k in range(num_pixels):
            for j in range(num_groups):
                if bin_images[i][k] == 1:
                    log_lam[j] += np.log(pi[j][k] + 1e-10)
                else:
                    log_lam[j] += np.log(1 - pi[j][k] + 1e-10)

        for j in range(num_groups):
            log_lam[j] += np.log(lam[j] + 1e-10)  # 加上先驗 log(π_j)

        # softmax 避免 overflow：log-sum-exp trick
        max_log = np.max(log_lam)
        log_lam -= max_log

        lam_copy = np.exp(log_lam)  # 還原成機率空間
        divide = np.sum(lam_copy)

        for j in range(num_groups):
            wi[i][j] = lam_copy[j] / (divide + 1e-10)  # 正規化成 posterior

    # M-step
    for j in range(num_groups):
        weight_sum = np.sum(wi[:, j])
        lam[j] = weight_sum / num_images

        for k in range(num_pixels):
            pi[j][k] = np.sum(wi[:, j] * bin_images[:, k]) / (weight_sum + 1e-10)

    # 收斂判斷（L1 norm）
    diff = np.sum(np.abs(pi - prev_pi))
    #print(f"L1 difference from previous pi: {diff:.6f}")

    # 每輪都重新 assign cluster → digit mapping（Hungarian）
    cost_matrix = np.zeros((num_groups, 10))
    for cluster_id in range(num_groups):
        # 從 wi 中先用 argmax 找出屬於所有照片的 cluster id
        # 根據 cluster_id 找出所有屬於這個 cluster 的圖片，用照片在 wi 中的 indices
        indices = np.where(np.argmax(wi, axis=1) == cluster_id)[0]
        # 根據這些 indices 找出這些圖片的 label
        true_labels = train_labels[indices]
        # 計算這個 cluster 中每個 digit 的數量
        for digit in range(10):
            cost_matrix[cluster_id][digit] = np.sum(true_labels == digit)
    # 用 Hungarian algorithm 找出 cluster 和 digit 的最佳 mapping
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_digit = {cluster: digit for cluster, digit in zip(row_ind, col_ind)}
    digit_to_cluster = {digit: cluster for cluster, digit in cluster_to_digit.items()}

    # 輸出當前 imagination（印到 terminal，也可以寫入檔案）
    #print(f"\n--- Imagination after Epoch {epochs} ---")
    for i in range(10):
        #print(pi[digit_to_cluster[i]])
        print(f"class {i}: ")
        for j in range(28):
            for k in range(28):
                if pi[digit_to_cluster[i]][j*28+k] > 0.5:
                    print("1", end="")
                else:
                    print("0", end="")
            print()
    print(f" No. of Iteration: {epochs}, Difference: {diff:.6f}")
    if diff < threshold:
        print("Converged!")
        break


print(wi[0])
# EM 預測的 cluster（0~9）
cluster_assign = np.argmax(wi, axis=1)  # 每張圖屬於哪個 cluster
print("Cluster assignment for first 10 images:", cluster_assign[:10])

# Use Hungarian algorithm to find the best mapping between clusters and digits
cost_matrix = np.zeros((num_groups, 10))
for cluster_id in range(num_groups):
    indices = np.where(cluster_assign == cluster_id)[0]
    true_labels = train_labels[indices]
    for digit in range(10):
        cost_matrix[cluster_id][digit] = np.sum(true_labels == digit)

row_ind, col_ind = linear_sum_assignment(-cost_matrix)
cluster_to_digit = {cluster: digit for cluster, digit in zip(row_ind, col_ind)}
digit_to_cluster = {digit: cluster for cluster, digit in cluster_to_digit.items()}

pred_path = 'result_plot/pred.txt'
with open(pred_path, 'w') as f:
    for i in range(10):
        #print(pi[digit_to_cluster[i]])
        print(f"Labeled class: {i}")
        f.write(f"Labeled class: {i}\n")
        for j in range(28):
            for k in range(28):
                if pi[digit_to_cluster[i]][j*28+k] > 0.5:
                    print("1", end="")
                    f.write("1")
                else:
                    print("0", end="")
                    f.write("0")
            print()
            f.write("\n")

# Confusion matrix
pred_digits = np.array([cluster_to_digit[c] for c in cluster_assign])
print("Predicted digits for first 10 images:", pred_digits[:10])
confusion_matrix(train_labels, pred_digits)

#  Iteration, Accuracy
print("Total iteration to converge:", epochs)
acc = accuracy_score(train_labels, pred_digits)
print("✅ Accuracy:", acc)

