from __future__ import division
import numpy as np
import scipy
import scipy.linalg
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
import metrics
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
from sklearn import preprocessing

def pseudoinverse_partition_indicator(H):
    """ function to compute the pseudoinverse H^+ of the partition indicator matrix H"""
    if not scipy.sparse.issparse(H):
        print("Partition Indicator matrix should be sparse matrix")

    # sparse matrices enjoy more pleasant syntax, so below is matrix mult.
    D = H.T*H
    Hplus = scipy.sparse.linalg.spsolve(D,H.T)
    return Hplus

def project_to(subspace_basis,vectors_to_project):
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
    X1 = S.T*V
    X2 = S.T*S
    projected = S*scipy.sparse.linalg.spsolve(X2,X1)

    return projected

def calculate_proj_error(U,V,norm):
    proj1 = project_to(U,V)

    if norm == 'F':
        error = scipy.linalg.norm(proj1)
    elif norm == '2':
        error = scipy.linalg.norm(proj1,2)

    return error

def test_random_projection(n=24,k=3,p=21):
    """Within an n dimensional space, project an orthogonal set of k vectors (a k-dimensional subspace) into a space of dimension p"""

    nsamples = 250
    # create another set of p random vectors
    U = ortho_group.rvs(dim=n)
    U = U[:,:p]

    error = np.zeros(nsamples)
    expected_error = np.sqrt(k*p/n)

    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:,:k]
        error[i] = calculate_proj_error(U,V,'F')
        print i, error[i]/expected_error

    plt.figure()
    plt.plot(error/expected_error)

    diff  = np.mean(error) - expected_error
    print diff
    print np.std(error)

    # second test
    error2 = np.zeros(nsamples)
    expected_error2 = np.sqrt(1 - k**2/n)
    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:,:k]
        clust = KMeans(n_clusters = k)
        clust.fit(V)
        partition_vec = clust.labels_
        partition_vec = spectral.relabel_partition_vec(partition_vec)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        H = preprocessing.normalize(H,axis=0,norm='l2')
        error2[i] = scipy.linalg.norm(V - project_to(H,V))


    plt.figure()
    plt.plot(error2/expected_error2)

    diff  = np.mean(error2) - expected_error2
    print diff
    print np.std(error2)


    return V, U
