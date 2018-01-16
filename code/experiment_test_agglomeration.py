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


def pseudoinverse_partition_indicator(H):
    """ function to compute the pseudoinverse H^+ of the partition indicator matrix H"""
    if not scipy.sparse.issparse(H):
        print("Partition Indicator matrix should be sparse matrix")

    # sparse matrices enjoy more pleasant syntax, so below is matrix mult.
    D = H.T*H
    Hplus = scipy.sparse.linalg.spsolve(D,H.T)
    return Hplus


def xlogy(x,y):
    """ compute x log(y) elementwise, with the convention that 0log0 = 0"""
    xlogy = x*np.log(y)
    xlogy[np.isinf(xlogy)] = 0
    xlogy[np.isnan(xlogy)] = 0
    return xlogy


def compute_likelihood_SBM(pvec,A,omega=None):
    H = spectral.create_partition_matrix_from_vector(pvec)
    # self-loops and directedness is not allowed here
    Emat, Nmat = spectral.compute_number_links_between_groups(A,pvec,directed=False)
    if omega is None:
        omega = Emat / Nmat

    logPmat = xlogy(Emat,omega) + xlogy(Nmat-Emat,1 - omega)
    likelihood = logPmat.sum()
    return likelihood

def project_orthogonal_to(subspace_basis,vectors_to_project):
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

    if not scipy.sparse.issparse(subspace_basis):
        S = np.matrix(subspace_basis)
    else:
        S = subspace_basis

    # print "HERE"
    # compute S*(S^T*S)^{-1}*S'*V
    X1 = S.T*V
    # print X1
    X2 = S.T*S
    # print X2
    projected = S*scipy.sparse.linalg.spsolve(X2,X1)
    # print projected

    orthogonal_proj = V - projected
    return orthogonal_proj

def calculate_proj_error(evecs,H,norm):
    n, k = np.shape(H)
    if n == k:
        error =0
        return error
    V = evecs[:,:k]
    proj1 = project_orthogonal_to(H,V)
    proj2 = project_orthogonal_to(V,H)


    if norm == 'F':
        norm1 = scipy.linalg.norm(proj1)/np.sqrt(k-k**2/n)
        norm2 = scipy.linalg.norm(proj2)/np.sqrt(k-k**2/n)
        error = 0.5*(norm1+norm2)
    elif norm == '2':
        norm1 = scipy.linalg.norm(proj1,2)
        norm2 = scipy.linalg.norm(proj2,2)
        error = .5*(norm1+norm2)

    return error

def test_agglomeration_ideas(groups_per_level=3):
    n=2*2700
    snr=5
    c_bar=20
    n_levels=3


    #generate
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1) # true partition lowest level
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    # display A
    plt.figure
    # plt.spy(A,markersize=0.01)
    plt.imshow(A.A,cmap='Greys')

    # do a first round of clustering with the Bethe Hessian
    p0 = spectral.cluster_with_BetheHessian(A,num_groups=groups_per_level**n_levels,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    p0 = spectral.relabel_partition_vec(p0)
    plt.figure()
    plt.plot(p0)


    # aggregate matrix
    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / Nagg
    plt.figure()
    plt.imshow(Aagg,interpolation='nearest')

    reg= False
    L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
    # D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
    # L = D - Aagg
    # plt.figure()
    # plt.imshow(L,interpolation='none')
    if reg:
        # set tau to average degree
        tau = Aagg.sum()/Aagg.shape[0]
    else:
        tau = 0

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    sigma = np.abs(ev[index[::-1]])
    # evecs = evecs[:,index]
    plt.figure()
    plt.plot(sigma)

    total_energy = sigma.sum()
    sigma_gap = np.abs(np.diff(sigma))
    approx_energy = np.cumsum(sigma)
    plt.figure()
    plt.plot(sigma_gap)
    plt.figure()
    plt.plot(np.diff(approx_energy/total_energy))

    max_k = np.max(p0)+1
    norm = 'F'
    mode = 'SBM'
    thres = 0.05
    error = np.zeros(max_k)
    likelihood = np.zeros(max_k)


    for k in range(max_k):
        partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
        print "K, error / exp rand error, likelihood"
        print k+1, error[k], likelihood[k]

    plt.figure()
    plt.plot(1+np.arange(max_k),error)
    plt.figure()
    plt.plot(1+np.arange(max_k),likelihood)
    return D_actual
