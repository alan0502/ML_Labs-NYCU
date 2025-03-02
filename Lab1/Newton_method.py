import numpy as np 
import LU as lu

def inverse_LU(A, degree):
    I = np.eye(degree)
    A_inv = np.zeros((degree, degree))
    L, U = lu.LU_decomposition(A)
    for i in range(degree):
        Y = lu.forward_substitution(L, I[:, i])
        X = lu.backward_substitution(U, Y)
        A_inv[:, i] = X
    return A_inv

def optimize(A, b, degree):
    X = np.zeros(degree)
    At_A = A.T @ A
    At_b = A.T @ b
    At_A_inv = inverse_LU(At_A, degree)
    while True:
        gradient = (At_A @ X) - At_b
        hassian = At_A_inv
        Xnew = X - hassian @ gradient
        if(np.linalg.norm(Xnew - X, ord=2) < 1e-6):
            break
        X = Xnew
    return X