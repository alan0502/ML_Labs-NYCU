import numpy as np
def cal_likelihood(data, target, num_bins):
    bin_size = 256 // num_bins # 計算 bin 大小
    #print(bin_size)
    image = data // bin_size
    #print(image[0])
    freq_table = np.zeros((10, 28, 28, num_bins), dtype=np.float32)  # ✅ 正確用法
    #print(data.shape)
    #print(image.shape)
    #print(target)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                freq_table[target[i]][j][k][image[i][j][k]] += 1
    #bin_index = pixel_value // 8  # 轉換為 bin
    #maximumlikelihood = data.groupby(target).size() / len(data)
    alpha = 1e-6
    for i in range(10):
        for j in range(28):
            for k in range(28):
                total_count = np.sum(freq_table[i][j][k])  # 類別 y 在 (j,k) 位置的總數
                freq_table[i, j, k, :] = (freq_table[i, j, k, :] + alpha) / (total_count + alpha * num_bins)

    return freq_table

def cal_posterior(train_data, train_target, test_data, num_bins, prior):
    freq_table = cal_likelihood(train_data, train_target, num_bins)
    print(freq_table[0][20][20])
    num_samples = test_data.shape[0]
    posterior_table = np.zeros((num_samples, 10), dtype=np.float32)
    bin_size = 256 // num_bins # 計算 bin 大小
    image = test_data // bin_size
    for i in range(num_samples):
        for y in range(10):
            log_sum = np.log(prior[y])
            for j in range(28):
                for k in range(28):
                    bin_idx = image[i, j, k]
                    log_sum += np.log(freq_table[y, j, k, bin_idx] + 1e-6)  # 避免 log(0)
            posterior_table[i, y] = log_sum
    print(posterior_table[0])
    #for i in range(num_samples):
    #    posterior_table[i] = np.exp(posterior_table[i] - np.max(posterior_table[i]))
    for i in range(num_samples):
        # 減去最大值，避免 overflow
        posterior_table[i] -= np.max(posterior_table[i])

        # 轉換回機率值
        posterior_table[i] = np.exp(posterior_table[i])

        # Marginalization: 確保機率總和為 1
        posterior_table[i] /= np.sum(posterior_table[i])

    return freq_table, posterior_table, np.argmax(posterior_table, axis=1)
    