import numpy as np
from numpy.linalg import svd

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

matr = np.matrix([[-1,+1,-1,+1],
                  [+1,-1,+1,-1],
                  [-1,+1,+0,+0],
                  [+0,+0,+1,-1],
                  [+1,-1,-1,+1]])

print(np.linalg.matrix_rank(matr))
print(nullspace(matr))