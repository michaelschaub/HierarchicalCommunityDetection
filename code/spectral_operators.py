#!/usr/bin/env python
"""
Spectral operators and functions spectral decomposition.
The module contains:


"""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, diags, issparse, csr_matrix


class SpectralOperator(object):

    def find_k_eigenvectors(self, K, which='SA'):
        M = self.operator
        if M.shape[0] == 1:
            print('WARNING: Matrix is a single element')
            evals = np.array([1])
            evecs = np.array([[1]])
        elif K < M.shape[0]:
            evals, evecs = eigsh(M, K, which=which)
        else:
            evals, evecs = eigsh(M, M.shape[0]-1, which=which)
        if which == 'SA':
            index = np.argsort(evals)
        elif which == 'SM':
            index = np.argsort(np.abs(evals))
        self.evals = evals[index]
        self.evecs = evecs[:, index]

    def find_negative_eigenvectors(self):
        """
        Find negative eigenvectors.

        Given a matrix M, find all the eigenvectors associated to negative
        eigenvalues and return number of negative eigenvalues
        """
        M = self.operator
        Kmax = M.shape[0] - 1
        K = min(10, Kmax)
        if self.evals is None:
            self.find_k_eigenvectors(K, which='SA')
        elif len(self.evals) < K:
            self.find_k_eigenvectors(K, which='SA')
        relevant_ev = np.nonzero(self.evals < 0)[0]
        while (relevant_ev.size == K):
            K = min(2 * K, Kmax)
            self.find_k_eigenvectors(K, which='SA')
            relevant_ev = np.nonzero(self.evals < 0)[0]
        self.evals = self.evals[relevant_ev]
        self.evecs = self.evecs[:, relevant_ev]
        return len(relevant_ev)


class Laplacian(SpectralOperator):

    def __init__(self, A):
        self.A = A
        self.build_operator()
        self.evals = None
        self.evecs = None

    def build_operator(self):
        """
        Construct a Laplacian matrix from the input matrix A. Output will be a
        sparse matrix or a dense matrix depending on input
        """
        A = self.A
        D = diags(np.ravel(A.sum(1)), 0)
        L = D - A

        self.operator = L


class BetheHessian(SpectralOperator):

    def __init__(self, A, regularizer='BHa'):

        self.A = A
        self.calc_r(regularizer)
        self.build_operator()
        self.evals = None
        self.evecs = None

    def calc_r(self, regularizer='BHa'):
        A = self.A
        if regularizer.startswith('BHa'):
            # set r to square root of average degree
            r = A.sum() / A.shape[0]
            r = np.sqrt(r)

        elif regularizer.startswith('BHm'):
            d = A.sum(axis=1).getA().flatten().astype(float)
            r = np.sum(d * d) / np.sum(d) - 1
            r = np.sqrt(r)

        # if last character is 'n' then use the negative version of the
        # BetheHessian
        if regularizer[-1] == 'n':
            self.r = -r
        else:
            self.r = r

    def build_operator(self):
        """
        Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
        B = (r^2-1)*I-r*A+D
        """
        A = self.A
        r = self.r
        A = test_sparse_and_transform(A)

        d = A.sum(axis=1).getA().flatten().astype(float)
        B = eye(A.shape[0]).dot(r**2 - 1) - r * A + diags(d, 0)
        self.operator = B


def test_sparse_and_transform(A):
    """ Check if matrix is sparse and if not, return it as sparse matrix"""
    if not issparse(A):
        print("""Input matrix not in sparse format,
                 transforming to sparse matrix""")
        A = csr_matrix(A)
    return A
