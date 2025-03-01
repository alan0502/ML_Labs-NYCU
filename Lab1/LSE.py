import numpy as np
import LU as lu

def regularization(A, b, degree, lam):
    AT_A = A.T @ A
    lamda_I = lam * np.eye(degree)
    AT_A_lam = AT_A + lamda_I
    AT_b = A.T @ b
    L, U = lu.LU_decomposition(AT_A_lam)
    z = lu.forward_substitution(L, AT_b) 
    w = lu.backward_substitution(U, z)
    #print(w)
    return w