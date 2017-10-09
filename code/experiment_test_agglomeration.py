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
    k = np.shape(H)[1]
    V = evecs[:,:k]
    proj1 = project_orthogonal_to(H,V)
    proj2 = project_orthogonal_to(V,H)


    if norm == 'F':
        norm1 = scipy.linalg.norm(proj1)/np.sqrt(k)
        norm2 = scipy.linalg.norm(proj2)/np.sqrt(k)
        error = 0.5*(norm1+norm2)
    elif norm == '2':
        norm1 = scipy.linalg.norm(proj1,2)
        norm2 = scipy.linalg.norm(proj2,2)
        error = .5*(norm1+norm2)

    # ~ print 'H\n',H.A
    # ~ print 'V\n',V
    #~ print proj1
    #~ print norm1,norm2,error
    #~ print '\n\n\n\n'

    return error


def test_agglomeration_ideas(groups_per_level=3):
    n=2700
    snr=5
    c_bar=20
    n_levels=3


    #generate
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    # display A
    # plt.figure()
    # plt.spy(A,markersize=1)

    # do a first round of clustering with the Bethe Hessian
    p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted',
                                            regularizer='BHa',clustermode='kmeans')
    # plt.figure()
    # plt.plot(p0)


    # aggregate matrix
    Aagg, _ = spectral.compute_number_links_between_groups(A,p0)
    # plt.figure()
    # plt.imshow(Aagg,interpolation='nearest')

    reg= False
    L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
    # D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
    # L = D - Aagg
    # plt.figure()
    # plt.imshow(L,interpolation='none')
    if reg:
        # set tau to average degree
        tau = Aagg.sum()/Aagmpog.shape[0]
    else:
        tau = 0

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    # evecs = evecs[:,index]
    plt.figure()
    plt.plot(ev[index])

    max_k = np.max(p0)+1
    norm = 'F'
    mode = 'SBM'
    thres = 0.05
    error = np.zeros(max_k)


    for k in range(1,max_k):
        partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
        # print "HNORM"
        # print Hnorm.A
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        print "K, error, error/max_k, error /k, error/sqrt(max_k), error/sqrt(k), thres "
        print k+1, error[k], error[k]/max_k, error[k]/k, error[k]/np.sqrt(max_k), error[k]/np.sqrt(k), thres

    plt.figure()
    plt.plot(1+np.arange(max_k),error)
