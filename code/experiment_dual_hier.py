from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, bmat


def createTiagoHierExpNetwork(n=100,a = 0.95, b = 0.75, c = 0.4, d = 0.1):
    
    A=coo_matrix(a*np.ones((n,n)))
    B=coo_matrix(b*np.ones((n,n)))
    C=coo_matrix(c*np.ones((n,n)))
    D=coo_matrix(d*np.ones((n,n)))
    
    E = bmat([[A, B, B, None, None, None, D, None, None, None, None, None],
                      [B, A, B, None, None, None, None, D, None, None, None, None],
                      [B, B, A, C, None, None, None, None, D, None, None, None],
                      [None, None, C, A, B, B, None, None, None, D, None, None],
                      [None, None, None, B, A, B, None, None, None, None, D, None],
                      [None, None, None, B, B, A, None, None, None, None, None, D],
                      [D, None, None, None, None, None, None, None, None, None, None, None],
                      [None, D, None, None, None, None, None, None, None, None, None, None],
                      [None, None, D, None, None, None, None, None, None, None, None, None],
                      [None, None, None, D, None, None, None, None, None, None, None, None],
                      [None, None, None, None, D, None, None, None, None, None, None, None],
                      [None, None, None, None, None, D, None, None, None, None, None, None]])
    
    
    return E.tocsr()
    