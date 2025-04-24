import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(w, X, y, eta, epoch):
    for i in range(epoch):
        w_prev = w.copy()
        z = X @ w  # shape: (2n, 1)
        #print(z)
        #print(z.shape)
        p = sigmoid(z)  # shape: (2n, 1)
        #print(p[:5])  # Print first 5 probabilities
        #print(p.shape)
        grad = X.T @ (p - y)
        w = w - eta * grad
        #if i % 10 == 0:
            #loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            #print(f'Epoch {i}: Loss = {loss:.4f}')
        if np.linalg.norm(w - w_prev) < 1e-3:
            break
        i += 1
    return w, p

def newtons_method(w, X, y, eta, epoch):
    for i in range(epoch):
        w_prev = w.copy()
        z = X @ w
        p = sigmoid(z)
        diag_values = (p * (1 - p)).flatten()
        D = np.diag(diag_values)  # shape: (n, n)
        H = X.T @ D @ X
        grad = X.T @ (p - y)
        if(np.linalg.det(H) < 1e-10):
            w = w - eta * grad
        else:
            w = w - np.linalg.inv(H) @ grad
        #if i % 10 == 0:
            #loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))
            #print(f'Epoch {i}: Loss = {loss:.4f}')
        if np.linalg.norm(w - w_prev) < 1e-3:
            break
        i += 1
    return w, p

def perceptron(w, X, y, eta, epoch):
    for i in range(epoch):
        w_prev = w.copy()
        for j in range(X.shape[0]):
            z = X[j] @ w
            zi = 1 if z >= 0 else 0  # Activation function
            if zi != y[j]:
                w = w + eta * (y[j] - zi) * X[j].reshape(-1, 1)
        if np.linalg.norm(w - w_prev) < 1e-3:
            break
        #if i % 10 == 0:
            #loss = np.mean(np.maximum(0, 1 - y * (X @ w)))
            #print(f'Epoch {i}: Loss = {loss:.4f}')
        i += 1
    p = np.where(X @ w >= 0, 1, 0)
    print(p)
    return w, p