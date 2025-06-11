import os
import numpy as np
import imageio.v2 as imread
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from dim_reduction import*

train_dir = "./Yale_Face_Database/Training"
test_dir = "./Yale_Face_Database/Testing"

def load_yale_faces(folder):
    X = []
    y = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pgm"):
            path = os.path.join(folder, fname)
            img = imageio.imread(path)
            X.append(img.flatten())
            label = int(fname.split('.')[0].replace("subject", ""))
            y.append(label)
            filenames.append(fname)
    return np.array(X), np.array(y), img.shape, filenames

def visualize_eigenfaces(eigenvectors, image_shape, title, n_rows=5, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        face = eigenvectors[:, i].reshape(image_shape)
        ax.imshow(face, cmap='gray')
        ax.set_title(f'PC {i+1}')
        ax.axis('off')
    plt.suptitle("Top 25 Eigenfaces", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"kernel_eigenface/eigenface/{title}.png", dpi=300)
    plt.show()

X_train, y_train, image_shape, filenames = load_yale_faces(train_dir)
X_test, y_test, _, _ = load_yale_faces(test_dir)
print("Train subjects:", sorted(set(y_train)))
print("Test subjects:", sorted(set(y_test)))
print(image_shape) # (231, 195)
#print(filenames)
print(X_train)
print(X_train.shape) # (135, 45045)，總共有 135 張圖片，每張圖片展平後有 45045 個像素，dimensions = 45045
#print(y_train)
#print(image_shape)

mode = int(input("Choose mode (0: PCA, 1: LDA): "))
if mode == 0:
    kernel = int(input("Kernel or not (0: No, 1: Yes): "))
    if kernel == 0:
        top_eigenvectors, mean_face = pca(X_train, n_components=25)
        visualize_eigenfaces(top_eigenvectors, image_shape, title="pca")
        pca_reconstruction(X_test, top_eigenvectors, mean_face, image_shape, num_samples=10)
        pca_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, mean_face, k=1)
    elif kernel == 1:
        kernel_mode = int(input("Choose kernel (0: Linear, 1: Polynomial, 2: RBF): "))
        if kernel_mode == 0:
            kernel_func = linear_kernel
        elif kernel_mode == 1:
            kernel_func = poly_kernel
        elif kernel_mode == 2:
            kernel_func = rbf_kernel
        else:
            raise ValueError("Invalid kernel mode selected.")
        top_eigenvectors, K_centered = pca_kernel(X_train, kernel_func, n_components=25)
        print(type(top_eigenvectors))
        print(top_eigenvectors.shape)
        #visualize_eigenfaces(top_eigenvectors, image_shape)
        acc = pca_kernel_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, K_centered, kernel_func, k=1)


elif mode == 1:
    # Step 1: Do PCA first
    pca_components, mean_face = pca(X_train, n_components=100)
    print("PCA components shape:", pca_components.shape)
    print("Mean face shape:", mean_face.shape)
    # Step 2: Project training data to PCA space
    X_pca = (X_train - mean_face) @ pca_components
    print("PCA projected shape:", X_pca.shape)
    # Step 3: Apply LDA in PCA-reduced space
    top_eigenvectors = lda(X_pca, y_train, n_components=25)  # now it's (100, 25)
    print("Top eigenvectors shape:", top_eigenvectors.shape)
    # Step 4: Fisherfaces = LDA projection back to original image space
    fisherfaces = top_eigenvectors.T @ pca_components.T  # shape: (25, 45045) # 25->45045 
    print("Fisherfaces shape:", fisherfaces.shape)
    # Step 5: Visualize
    visualize_eigenfaces(fisherfaces.T, image_shape, title="lda")
    lda_recognition(X_train, y_train, X_test, y_test, top_eigenvectors, mean_face, fisherfaces, k=1)
    lda_reconstruction(X_test, mean_face, pca_components, top_eigenvectors, image_shape, num_samples=10)
