import numpy as np

def LU_decomposition(A):
    U = A.copy()
    L = np.eye(len(U))
    for i in range(len(A)-1):
        lead = U[i][i]
        if lead == 0:
            raise ValueError("Zero pivot encountered, consider using partial pivoting.")
        for j in range(i+1, len(U)):
            power = U[j][i]/lead
            L[j][i] = power
            U[j, i:] -= power * U[i, i:]
    return L, U

def forward_substitution(L, b):
    lens = len(b)
    z = np.zeros(lens)
    for i in range(lens):
        z[i] = (b[i] - np.dot(L[i][:i], z[:i]))/L[i][i]
    return z

def backward_substitution(U, z):
    lens = len(z)
    w = np.zeros(lens)
    for i in range (lens-1, -1, -1):
        w[i] = (z[i] - np.dot(U[i][i+1:], w[i+1:]))/U[i][i]
    return w