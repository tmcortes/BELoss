import numpy as np

def whitenapply(X, m, P, dimensions=None):
    
    # X is [DxN] (N feature vectors D dimensional as columns)
    # m is [Dx1]
    # P is [DxD] (you can pick the first D' dimensions)
    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X

def pcawhitenlearn(X):

    # X is [DxN] (N feature vectors D dimensional as columns)
    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    
    return m, P

def whitenlearn(X, qidxs, pidxs):

    # X is [DxN] (N feature vectors D dimensional as columns)
    # m is [Dx1]
    # P is [DxD] (you can pick the first D' dimensions)
    
    # Learning Lw w annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]
    try:
        P = np.linalg.inv(np.linalg.cholesky(S))
    except:
        P = np.eye(S.shape[0])
    df = np.dot(P, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P)

    return m, P

def mahadist( x, y, cov ):
    
    r = (x - y)[np.newaxis]
    S = np.linalg.inv(cov)
    return np.sqrt( np.dot( np.dot(r, S), r.T ) )

def eucldist( x, y ):
    
    r = (x - y)[np.newaxis]
    return np.sqrt(np.dot( r, r.T))

#X = np.array([[10, 500],
#              [15, 500],
#              [10, 550],
#              [30, 500],
#              [10, 700],
#              [30, 500],
#              [12, 550],
#              [15, 700],
#              [20, 1000],
#              [17, 800],
#              [16, 750]])
#  
#Sigma = np.cov(X.T, ddof=0)
#L, Q = np.linalg.eig(Sigma)
#L = np.diag(L)
#invL = np.linalg.inv(L)
#invL_sq = np.linalg.inv(np.sqrt(L))
#invSigma = Q.dot(invL).dot(Q.T)
#P = invL_sq.dot(Q.T)
#
#x = X[0][np.newaxis]
#y = X[3][np.newaxis]
#r = (x - y )
#
#a = r.dot(invSigma).dot(r.T)
#b = r.dot(Q).dot(invL).dot(Q.T).dot(r.T)
#c = r.dot(Q).dot(invL_sq).dot(invL_sq).dot(Q.T).dot(r.T)
#Px = P.dot(x)
#Py = P.dot(y)
#d = (Px-Py).T.dot(Px-Py)




