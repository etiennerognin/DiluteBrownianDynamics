import numpy as np

def rotateRandomly(A):
    """Rotate matrix A or list of matrices with same rotation. 
    
    Parameters
    ----------
    A : (9,) ndarray or list of (9,) ndarray
        Matrix/matrices to rotate. A is in line format.
    
    Returns
    -------
    (9,) ndarray or list of (9,) ndarray
        Input rotated by a random rotation matrix.
    
    Notes
    ----- 
    Scipy module to achieve this not working. 
    """
    x = 2*np.pi*np.random.random()
    y = 2*np.pi*np.random.random()
    z = 2*np.pi*np.random.random()
    Rx = np.array([[1., 0, 0],
                  [0, np.cos(x), -np.sin(x) ],
                  [0, np.sin(x), np.cos(x)]
                  ])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                  [0, 1, 0],
                  [-np.sin(y), 0, np.cos(y)]
                  ])
    Rz = np.array([[np.cos(z),  -np.sin(z), 0],
                  [np.sin(z), np.cos(z), 0],
                  [0, 0, 1]
                  ])
    R = np.matmul(np.matmul(Rx, Ry),Rz)
    if A is list or tuple:
        out = []
        for matrix in A:
            matrix = np.reshape(matrix, (3,3))
            matrix = np.matmul(np.matmul(R,matrix), np.transpose(R))
            matrix = np.reshape(matrix, (9))
            out.append(matrix)
        return out    
    else:
        Ar = np.reshape(A, (3,3))
        Ar = np.matmul(np.matmul(R,Ar), np.transpose(R))
        return np.reshape(Ar, (9))


def getWi(U):
    """Return Weissenberg number of velocity gradient U. Defined as largest eigenvalue of symmetric part of U."""
    U_mat = np.reshape(U,(3,3))
    D = 0.5*(U_mat + U_mat.T)
    return np.linalg.eigvalsh(D)[-1]
    
def random_gradU():
    U_plus = np.random.standard_normal(9)/np.sqrt(2)
    trU = U_plus[0] + U_plus[4] + U_plus[8]
    U_plus += -trU/3*np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.])
    return U_plus
