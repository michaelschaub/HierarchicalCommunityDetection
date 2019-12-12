from __future__ import division
import numpy as np
import scipy
import scipy.linalg
import GHRGbuild
import spectral_algorithms_new as spectral

from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
from sklearn import preprocessing

def project_to(subspace_basis, vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space to which we want to project
    vectors_to_project: project these vectors into the specified subspace
    """

    if not scipy.sparse.issparse(vectors_to_project):
        V = np.matrix(vectors_to_project)
    else:
        V = vectors_to_project

    if not scipy.sparse.issparse(subspace_basis):
        S = np.matrix(subspace_basis)
    else:
        S = subspace_basis

    # compute S*(S^T*S)^{-1}*S'*V
    X1 = S.T * V
    X2 = S.T * S
    projected = S * scipy.linalg.solve(X2, X1)

    return projected

def project_orthogonal_to(subspace_basis, vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space orthogonal to which we want to project
    vectors_to_project: project these vectors into the orthogonal complement of the
    specified subspace
    """

    if not scipy.sparse.issparse(vectors_to_project):
        V = np.matrix(vectors_to_project)
    else:
        V = vectors_to_project

    orthogonal_proj = V - project_to(subspace_basis, V)

    return orthogonal_proj

def calculate_proj_error(U, V, norm):
    proj1 = project_orthogonal_to(U, V)

    if norm == 'F':
        error = scipy.linalg.norm(proj1)**2
    elif norm == '2':
        error = scipy.linalg.norm(proj1, 2)

    return error

def test_random_projection(n=10):
    """Within an n dimensional space, project an orthogonal set of k vectors (a k-dimensional subspace) into a space of dimension n-k
    where 1 vector is shared"""
    plt.close('all')

    nsamples = 1000

    testdim = np.arange(1,n+1)
    print testdim
    meanerror = np.zeros(testdim.size)
    stderror = np.zeros(testdim.size)

    for (j, k) in enumerate(testdim):
        # create an orthogonal set of random vectors
        U = ortho_group.rvs(dim=n)
        U2 = U[:, :k]
        error = np.zeros(nsamples)
        if k<=1:
            continue
        for i in range(nsamples):
            Q = ortho_group.rvs(dim=n-1)
            V = U[:,1:].dot(Q)
            V2 = np.hstack((U[:,0:1],V))[:,:k]
            error[i] = calculate_proj_error(U2, V2, 'F')
        meanerror[j] = np.mean(error)
        stderror[j] = np.std(error)

    expected_error2 = (testdim-1) * (n-testdim) /(n-1)
    plt.figure()
    plt.errorbar(testdim, meanerror, stderror)
    plt.errorbar(testdim, expected_error2)
    plt.show()

def test_random_projection2(n=10):
    """Within an n dimensional space, project an orthogonal set of k vectors (a k-dimensional subspace) into a space of dimension n-k
    where 1 vector is shared
    Hierarchical version: make the two subspaces align at the first vector and the fourth
    """
    plt.close('all')

    nsamples = 1000

    testdim = np.arange(1,n+1)
    print testdim
    meanerror = np.zeros(testdim.size)
    stderror = np.zeros(testdim.size)

    for (j, k) in enumerate(testdim):
        # create an orthogonal set of random vectors
        U = ortho_group.rvs(dim=n)
        U2 = U[:, :k]
        error = np.zeros(nsamples)
        if k<=1:
            continue
        for i in range(nsamples):
            Q = ortho_group.rvs(dim=2)
            V1 = U[:,1:3].dot(Q)
            Q = ortho_group.rvs(dim=n-3)
            Vend = U[:,3:].dot(Q)
            V2 = np.hstack((U[:,0:1],V1,Vend))[:,:k]
            error[i] = calculate_proj_error(U2, V2, 'F')
        meanerror[j] = np.mean(error)
        stderror[j] = np.std(error)

    levels = [1,3,n]
    start_end_pairs = zip(levels[:-1],levels[1:])

    expected_error = []
    for i,j in start_end_pairs:
        Ks = np.arange(i,j)
        errors = (Ks - i) * (j - Ks) / (j-i)
        expected_error = np.hstack([expected_error,errors])
    expected_error2 = np.hstack([expected_error,0])

    plt.figure()
    plt.errorbar(testdim, meanerror, stderror)
    plt.errorbar(testdim, expected_error2)
    plt.show()