import numpy as np
import struct
import matplotlib.pyplot as plt
import NBC_discrete as nbc_dis
import NBC_continuous as nbc_con

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

def compute_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

# Read MNIST data
train_images = load_mnist_images("inputimage/train-images.idx3-ubyte__")
train_labels = load_mnist_labels("inputimage/train-labels.idx1-ubyte__")
test_images = load_mnist_images("inputimage/t10k-images.idx3-ubyte__")
test_labels = load_mnist_labels("inputimage/t10k-labels.idx1-ubyte__")
unique_labels = np.unique(train_labels)
#print(f"Train images shape: {train_images.shape}")  # (60000, 28, 28)
#print(f"Train labels shape: {train_labels.shape}")  # (60000,)
#print(f"Test images shape: {test_images.shape}")  # (10000, 28, 28)
#print(f"Test labels shape: {test_labels.shape}")  # (10000,)
#print(train_labels.shape[0])
#print(train_images[0])

count = np.bincount(train_labels) 
#print(count) 
prior = count / len(train_labels)  

num_bins = 32

mode = int(input("Toggle option (0: discrete/1: continuous): "))
if mode == 0:
    print("Using Discrete Mode")
    freq_table, posterior_table, pred_labels = nbc_dis.cal_posterior(train_images, train_labels, test_images, num_bins, prior)
    pred_path = 'outputfile/pred_discrete.txt'
    imagine_path = 'outputfile/imagine_discrete.txt'
    accuracy = compute_accuracy(pred_labels, test_labels)
    with open(pred_path, 'w') as f:
        for i in range(10000):
            #f.write(f"Posterior (in log scale):\n")
            print("Posterior (in log scale):")
            for y in range(10):
                #f.write(f"{y}: {posterior_table[i, y]:.15f}\n")
                print(f"{y}: {posterior_table[i, y]:.15f}")
            #f.write(f"Prediction: {pred_labels[i]}, Ans: {test_labels[i]}\n")
            print(f"Prediction: {pred_labels[i]}, Ans: {test_labels[i]}")
            #f.write("\n")
            print()
        #f.write(f"Naive Bayes Accuracy: {accuracy:.2f}%\n")
        print(f"Naive Bayes Accuracy: {accuracy:.2f}%")
    print()
    with open(imagine_path, 'w') as f:
        #f.write("Imagination of numbers in Bayesian classifier:\n")
        print("Imagination of numbers in Bayesian classifier:")
        for i in range(10):
            #f.write(f"{i}:\n")
            print(f"{i}:")
            for j in range(28):
                for k in range(28):
                    max_bin = np.argmax(freq_table[i][j][k])
                    if(max_bin < 16):
                        #f.write("0")
                        print("0", end="")
                    else:
                        #f.write("1")
                        print("1", end="")
                #f.write("\n")
                print()
            #f.write("\n")
            print()
    #print_posterior(nbc, test_labels, num_samples=5)
    #print(nbc[0][20][20])
elif mode == 1:
    print("Using Continuous Mode")
    posterior_table, pred_labels, mean = nbc_con.cal_posterior(train_images, train_labels, test_images, prior)
    print(posterior_table[0])
    print(pred_labels[0])
    pred_path = 'outputfile/pred_continuous.txt'
    imagine_path = 'outputfile/imagine_continuous.txt'
    accuracy = compute_accuracy(pred_labels, test_labels)
    print(f"Naive Bayes Accuracy: {accuracy:.2f}%")
    with open(pred_path, 'w') as f:
        for i in range(10000):
            #f.write(f"Posterior (in log scale):\n")
            print("Posterior (in log scale):")
            for y in range(10):
                #f.write(f"{y}: {posterior_table[i, y]:.15f}\n")
                print(f"{y}: {posterior_table[i, y]:.15f}")
            #f.write(f"Prediction: {pred_labels[i]}, Ans: {test_labels[i]}\n")
            print(f"Prediction: {pred_labels[i]}, Ans: {test_labels[i]}")
            #f.write("\n")
            print()
        #f.write(f"Naive Bayes Accuracy: {accuracy:.2f}%\n")
        print(f"Naive Bayes Accuracy: {accuracy:.2f}%")
    print()
    with open(imagine_path, 'w') as f:
        #f.write("Imagination of numbers in Bayesian classifier:\n")
        print("Imagination of numbers in Bayesian classifier:")
        for i in range(10):
            #f.write(f"{i}:\n")
            print(f"{i}:")
            for j in range(28):
                for k in range(28):
                    if mean[i][j][k] > 128:
                        #f.write("1")
                        print("0", end="")
                    else:
                        #f.write("0")
                        print("1", end="")
                #f.write("\n")
                print()
            #f.write("\n")
            print()
    print()
