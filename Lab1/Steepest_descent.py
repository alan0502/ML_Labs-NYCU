import numpy as np

def descent(A, b, degree, lam, lr):
    X = np.random.rand(degree)
    # X = np.zeros(degree)
    At_b = A.T @ b
    AT_A = A.T @ A
    n = 100000
    gamma = 0.01
    while n > 0:
        gradient = 2*(AT_A @ X) - 2*At_b + lam*np.sign(X)
        #eta = lr * np.exp(-gamma * (100000-n))
        Xnew = X - lr * gradient
        #print(np.linalg.norm(Xnew - X, ord=2))
        X = Xnew
        n = n-1
    return X