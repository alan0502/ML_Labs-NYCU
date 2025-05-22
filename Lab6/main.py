from load_image import*
from cluster import*
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

def visualize_segmentation(labels, image_size):
    segmented = labels.reshape(image_size)
    #plt.imshow(segmented, cmap='gray')
    plt.imshow(segmented, cmap='tab10')
    plt.title("Kernel K-means Segmentation")
    plt.axis('off')
    plt.show()

def visualize_loss(losses, dir_name, img_name, title):
    epochs = range(1, len(losses)+1)
    plt.plot(epochs, losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{dir_name}/{img_name}')
    plt.show()

def compute_kernel(feature, gamma_s, gamma_c):
    # Extract coordinates and colors
    coords = feature[:, :2]
    colors = feature[:, 2:]
    # Compute pairwise squared differences in spatial coordinates
    diff_s = cdist(coords, coords, metric='sqeuclidean')
    # Compute pairwise squared differences in color space
    diff_c = cdist(colors, colors, metric='sqeuclidean')
    # Compute the kernel matrix
    kernel_matrix = np.exp(-gamma_s * diff_s) * np.exp(-gamma_c * diff_c)
    return kernel_matrix

# Load images and convert to features
feature1, size1 = load_image_as_features("images/image1.png")
feature2, size2 = load_image_as_features("images/image2.png")
#print("Feature1 shape:", feature1.shape) #(10000, 5)
#print("Image1 size:", size1) # (100, 100)
#print("Feature2 shape:", feature2.shape) #(10000, 5)
#print("Image2 size:", size2) # (100, 100)

kernel1 = compute_kernel(feature1, gamma_s=1e-4, gamma_c=1e-4)
D1 = np.diag(kernel1.sum(axis=1))
kernel2 = compute_kernel(feature2, gamma_s=1e-4, gamma_c=1e-4)
D2 = np.diag(kernel2.sum(axis=1))
#print(kernel1[1, 2])
#print(kernel1.shape) # (10000, 10000)
mode = int(input("Mode (0: Kernel k-means, 1: Spectral ratio cut, 2: Spectral normalized cut): "))
if mode == 0:
    k_num = int(input("k: "))
    kmeanspp = int(input("Initialize Label (0: random, 1: k-means++): "))

    if kmeanspp == 0:
        dir_name = f"kernel_kmeans/image1/k={k_num}"
    else:
        dir_name = f"kernel_kmeans/image1++/k={k_num}"
    labels1, loss1 = kernel_kmeans(epoch=30, k=k_num, kernel=kernel1, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss1, dir_name, "loss", "Kernel K-means Loss Over Epochs")
    visualize_segmentation(labels1, size1)

    if kmeanspp == 0:
        dir_name = f"kernel_kmeans/image2/k={k_num}"
    else:
        dir_name = f"kernel_kmeans/image2++/k={k_num}"
    labels2, loss2 = kernel_kmeans(epoch=30, k=k_num, kernel=kernel2, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss2, dir_name, "loss", "Kernel K-means Loss Over Epochs")
    visualize_segmentation(labels2, size2)
    
elif mode == 1:
    k_num = int(input("k: "))
    kmeanspp = int(input("Initialize Label (0: random, 1: k-means++): "))

    if kmeanspp == 0:
        dir_name = f"spectral/ratio/image1/k={k_num}"
    else:
        dir_name = f"spectral/ratio/image1++/k={k_num}"
    labels1, loss1 = spectral_ratio_cut(W=kernel1, epoch=20, k=k_num, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss1, dir_name, "loss", "Spectral Ratio Cut Loss Over Epochs")
    visualize_segmentation(labels1, size1)

    if kmeanspp == 0:
        dir_name = f"spectral/ratio/image2/k={k_num}"
    else:
        dir_name = f"spectral/ratio/image2++/k={k_num}"
    labels2, loss2 = spectral_ratio_cut(W=kernel2, epoch=20, k=k_num, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss2, dir_name, "loss", "Spectral Ratio Cut Loss Over Epochs")
    visualize_segmentation(labels2, size2)

elif mode == 2:
    k_num = int(input("k: "))
    kmeanspp = int(input("Initialize Label (0: random, 1: k-means++): "))

    if kmeanspp == 0:
        dir_name = f"spectral/normalized/image1/k={k_num}"
    else:
        dir_name = f"spectral/normalized/image1++/k={k_num}"
    labels1, loss1 = spectral_normalized_cut(W=kernel1, epoch=20, k=k_num, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss1, dir_name, "loss", "Spectral Normalized Cut Loss Over Epochs")
    visualize_segmentation(labels1, size1)

    if kmeanspp == 0:
        dir_name = f"spectral/normalized/image2/k={k_num}"
    else:
        dir_name = f"spectral/normalized/image2++/k={k_num}"
    labels2, loss2 = spectral_normalized_cut(W=kernel2, epoch=20, k=k_num, image_size=(100, 100), dir_name=dir_name, kmeanspp=kmeanspp)
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    visualize_loss(loss2, dir_name, "loss", "Spectral Normalized Cut Loss Over Epochs")
    visualize_segmentation(labels2, size2)

else:
    dir_name = f"spectral/ratio/image2++/k=3"
    make_gif_from_frames(dir_name, f"{dir_name}.gif", duration=500)
    print("Invalid Mode")





