import numpy as np

def cal_posterior(train_images, train_labels, test_data, prior):
    #print("cal_posterior")
    mean = np.zeros((10, 28, 28), dtype=np.float32)
    var = np.zeros((10, 28, 28), dtype=np.float32)
    #print(train_images)
    for i in range(10):  
        mean[i] = np.mean(train_images[train_labels == i], axis=0)  
        var[i] = np.var(train_images[train_labels == i], axis=0) + 1e-6 
    #print(mean[0])
    #print(var[0])
    var = var**(1/1000)
    #print(test_data) (10000, 28, 28)
    num_samples = test_data.shape[0]
    posterior_table = np.zeros((num_samples, 10), dtype=np.float32)
    for y in range(10):
        # calculate log P(X | y) 
        log_likelihood = -0.5 * np.sum(((test_data - mean[y]) ** 2) / var[y] + np.log(2 * np.pi * var[y]), axis=(1, 2))
        # calculate log P(y | X)
        posterior_table[:, y] = np.log(prior[y]) + log_likelihood
    for i in range(num_samples):
        # 減去最大值，避免 overflow
        posterior_table[i] -= np.max(posterior_table[i])

        # 轉換回機率值
        posterior_table[i] = np.exp(posterior_table[i])

        # Marginalization: 確保機率總和為 1
        posterior_table[i] /= np.sum(posterior_table[i])
    return posterior_table, np.argmax(posterior_table, axis=1), mean