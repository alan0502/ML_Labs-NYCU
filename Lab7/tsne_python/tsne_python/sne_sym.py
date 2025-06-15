import numpy as np
import pylab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import os

def plot_similarity_distribution(P, Q, method_name):
    # 移除對角線（p_ii, q_ii = 0）
    n = P.shape[0]
    P_flat = P[np.triu_indices(n, k=1)]
    Q_flat = Q[np.triu_indices(n, k=1)]

    # 畫出 histogram
    plt.figure(figsize=(8, 4))
    plt.hist(P_flat, bins=100, alpha=0.5, label="P (High-D)", density=True)
    plt.hist(Q_flat, bins=100, alpha=0.5, label="Q (Low-D)", density=True)
    plt.yscale('log')
    plt.xlabel("Similarity")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Similarity Distribution - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"result/similarity_dist_{method_name}.png", dpi=300)
    #plt.show()
    plt.close()

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P # H: Shannon entropy, P: probability distribution (P-row)


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, labels=None, save_gif=False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.	# early exaggeration
    P = np.maximum(P, 1e-12)

    # Initialize error tracking
    errors = []
    frames = []  # 🆕 儲存每 N 輪嵌入狀態
    frame_interval = 20  # 每幾輪紀錄一次圖

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        #num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            errors.append(C)
            print("Iteration %d: error is %f" % (iter + 1, C))

        if save_gif and (iter % frame_interval == 0 or iter == max_iter - 1):
            fig, ax = plt.subplots()
            if labels is not None:
                scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=5)
            else:
                scatter = ax.scatter(Y[:, 0], Y[:, 1], s=5)
            ax.set_title(f"Iter {iter}")
            ax.axis("off")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)
            plt.close(fig)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    os.makedirs("result", exist_ok=True)
    if save_gif and frames:
        gif_path = "result/SNE_sym_opt.gif"
        imageio.mimsave(gif_path, frames, duration=0.3)
        print(f"🎬 Optimization GIF saved to: {gif_path}")

    # Plot error curve
    plt.plot(np.arange(10, max_iter + 1, 10), errors)
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence (Cost)")
    plt.title("SNE_sym Optimization Error Curve")
    plt.grid(True)
    plt.savefig("result/SNE_sym_curve.png", dpi=300)
    #plt.show()
    plt.close()

    plot_similarity_distribution(P, Q, method_name="SNE_sym")

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform SNE_sym on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 10.0, labels=labels, save_gif=True)
    plt.axis('equal')               
    plt.xlim(-110, 110)             
    plt.ylim(-110, 110)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig("result/SNE_sym.png", dpi=300)
    #pylab.show()
    pylab.close()
