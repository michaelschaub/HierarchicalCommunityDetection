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
from helperfunctions import *

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

    orthogonal_proj = V - project_to(subspace_basis,V)

    return orthogonal_proj

def calculate_proj_error(U,V,norm):
    proj1 = project_to(U,V)

    if norm == 'F':
        error = scipy.linalg.norm(proj1)
    elif norm == '2':
        error = scipy.linalg.norm(proj1,2)

    return error

def calculate_proj_error2(evecs,H,norm):
    n, k = np.shape(H)
    if n == k:
        error =0
        return error
    V = evecs[:,:k]
    proj1 = project_orthogonal_to(H,V)
    proj2 = project_orthogonal_to(V,H)


    if norm == 'F':
        norm1 = scipy.linalg.norm(proj1)
        norm2 = scipy.linalg.norm(proj2)
        error = 0.5*(norm1+norm2)
    elif norm == '2':
        norm1 = scipy.linalg.norm(proj1,2)
        norm2 = scipy.linalg.norm(proj2,2)
        error = .5*(norm1+norm2)

    return error

def calculate_proj_error3(evecs,H,norm):
    n, k = np.shape(H)
    if n == k:
        error =0
        return error
    V = evecs[:,:k]
    proj1 = project_orthogonal_to(H,V)


    if norm == 'F':
        error = scipy.linalg.norm(proj1)
    elif norm == '2':
        norm1 = scipy.linalg.norm(proj1,2)
        norm2 = scipy.linalg.norm(proj2,2)
        error = .5*(norm1+norm2)

    return error

def test_random_projection(n=24,p=12):
    """Within an n dimensional space, project an orthogonal set of k vectors (a k-dimensional subspace) into a space of dimension p"""

    nsamples = 1000
    # create another set of p random vectors
    U = ortho_group.rvs(dim=n)
    U = U[:,:p]


    testdim =np.arange(2,p+1)
    meanerror = np.zeros(testdim.size)
    stderror = np.zeros(testdim.size)
    meanerror2 = np.zeros(testdim.size)
    stderror2 = np.zeros(testdim.size)
    for (j,k) in enumerate(testdim):
        error = np.zeros(nsamples)
        error2 = np.zeros(nsamples)
        expected_error = np.sqrt(k*p/n)
        expected_error2 = np.sqrt((n-p)*k/n)
        for i in range(nsamples):
            # create k orthogonal vectors
            V = ortho_group.rvs(dim=n)
            V = V[:,:k]
            error[i] = calculate_proj_error(U,V,'F')
            error2[i] = scipy.linalg.norm(project_orthogonal_to(U,V))
        meanerror[j] = np.mean(error)-expected_error
        stderror[j] = np.std(error)
        meanerror2[j] = np.mean(error2)-expected_error2
        stderror2[j] = np.std(error2)

    plt.figure()
    plt.errorbar(testdim,meanerror,stderror)
    plt.figure()
    plt.errorbar(testdim,meanerror2,stderror2)


def test_random_projection_with_kmeans(n=3):
    nsamples = 20

    # second test
    testdim =np.arange(2,n+1)
    meanerror = np.zeros(testdim.size)
    stderror = np.zeros(testdim.size)
    meanerror2 = np.zeros(testdim.size)
    stderror2 = np.zeros(testdim.size)
    for (j,k) in enumerate(testdim):
        error = np.zeros(nsamples)
        error2 = np.zeros(nsamples)
        expected_error = np.sqrt(k-k**2/n)
        clust = KMeans(n_clusters = k)
        for i in range(nsamples):
            # create k orthogonal vectors
            V = ortho_group.rvs(dim=n)
            V = V[:,:k]
            # H = ortho_group.rvs(dim=n)
            # H = H[:,:k]
            clust.fit(V)
            partition_vec = clust.labels_
            partition_vec = spectral.relabel_partition_vec(partition_vec)
            H = spectral.create_partition_matrix_from_vector(partition_vec)
            H = preprocessing.normalize(H,axis=0,norm='l2')
            error[i] = scipy.linalg.norm(project_orthogonal_to(H,V))
            error2[i] = error[i] - expected_error
        meanerror[j] = np.mean(error)
        stderror[j] = np.std(error)
        meanerror2[j] = np.mean(error2)
        stderror2[j] = np.std(error2)

    plt.figure()
    plt.errorbar(testdim,meanerror,stderror)
    plt.figure()
    plt.errorbar(testdim,meanerror2,stderror2)

def test_random_projection_with_kmeans2(n,k):
    nsamples = 20

    if n==k:
        return 0, 0

    error = np.zeros(nsamples)
    clust = KMeans(n_clusters = k)
    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:,:k]
        clust.fit(V)
        partition_vec = clust.labels_
        partition_vec = spectral.relabel_partition_vec(partition_vec)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        H = preprocessing.normalize(H,axis=0,norm='l2')
        error[i] = scipy.linalg.norm(project_orthogonal_to(H,V))
    meanerror = np.mean(error)
    stderror = np.std(error)

    return meanerror, stderror


def test_projection_diagonal_blocks(n=9,q=4):

    nsamples = 20
    p =0.5
    p2 = 0.2
    error = np.zeros(nsamples)
    error2 = np.zeros(nsamples)
    norm = "F"
    mode = "SBM"


    ###############################
    # CREATE org. graph and cluster
    ##############################
    Aorg = create_diagonal_graph(n=n,prob=p,prob2=p2)
    Aorg = spectral.add_noise_to_small_matrix(Aorg)
    # print Aorg
    L = spectral.construct_graph_Laplacian(Aorg)
    ev, evecs = scipy.linalg.eigh(L)
    print ev
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index]
    tau = None
    Dtau_sqrt_inv = None
    pvec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv)
    error_orig = calculate_proj_error2(evecs, Hnorm, norm)
    print relabel_partition_vec(pvec)
    print error_orig


    ###################################
    # TEST what happens if we add noise
    ###################################
    for ii in range(nsamples):
        A = spectral.add_noise_to_small_matrix(Aorg)
        L = spectral.construct_graph_Laplacian(A)

        ev, evecs = scipy.linalg.eigh(L)
        index = np.argsort(np.abs(ev))
        evecs = evecs[:,index]

        tau = None
        Dtau_sqrt_inv = None
        partition_vec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv,method='QR')
        error[ii] = calculate_proj_error2(evecs, Hnorm, norm)

        partition_vec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv, method='KM')
        error2[ii] = calculate_proj_error2(evecs, Hnorm, norm)

    plt.figure()
    plt.plot(error)
    plt.figure()
    plt.plot(error2)

    ################################################################
    # CREATE graph averaged according to orig partitiion and cluster
    ################################################################
    Aorg_av = create_averaged_graph(A,pvec)
    # print Aorg_av
    L = spectral.construct_graph_Laplacian(Aorg_av)
    ev, evecs = scipy.linalg.eigh(L)
    print ev
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index]
    tau = None
    Dtau_sqrt_inv = None
    pvec2, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv)
    error_orig = calculate_proj_error2(evecs, Hnorm, norm)
    print relabel_partition_vec(pvec2)
    print error_orig

    #####################################################
    # TEST what happens if we add noise to averaged graph
    #####################################################
    for ii in range(nsamples):
        A = spectral.add_noise_to_small_matrix(Aorg_av)
        L = spectral.construct_graph_Laplacian(A)

        ev, evecs = scipy.linalg.eigh(L)
        index = np.argsort(np.abs(ev))
        evecs = evecs[:,index]

        tau = None
        Dtau_sqrt_inv = None
        partition_vec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv,method='QR')
        error[ii] = calculate_proj_error2(evecs, Hnorm, norm)

        partition_vec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv, method='KM')
        error2[ii] = calculate_proj_error2(evecs, Hnorm, norm)
    plt.figure()
    plt.plot(error)
    plt.figure()
    plt.plot(error2)


    meanerror, stderror = test_random_projection_with_kmeans2(27,q)
    print meanerror, stderror
    print np.sqrt(q-q**2/27)

def create_diagonal_graph(n=9,prob=0.5,prob2=0.3):
    A = (prob-prob2)*np.diag(np.ones(n)) + (prob-prob2)*np.ones((n,n))
    return A


def create_agglomerated_graphGHRH():

    groups_per_level=3
    n_levels=4
    n=3**10
    c_bar=50
    snr = 7
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    #get true hierarchy
    true_pvec = D_actual.get_partition_all()[-1]
    Eagg, Nagg = compute_number_links_between_groups(A,true_pvec)
    Aagg = Eagg / Nagg

    return Aagg


def create_averaged_graph(A,pvec):
    H = spectral.create_partition_matrix_from_vector(pvec)
    Hplus = pseudoinverse_partition_indicator(H)
    A = np.matrix(A)
    A_av = H*Hplus*A*Hplus.T*H.T
    return A_av

def test_kmeans_bootstrap_comparison():
    Aorg = create_agglomerated_graphGHRH()
    n, _ = Aorg.shape

    ###################################
    # TEST what happens if we add noise
    ###################################
    Ks = np.arange(n,1,-1)
    print Ks
    print Ks.shape
    errors, std_errors = identify_next_level(Aorg,Ks)
    errors2 = np.zeros(n)
    print errors.shape
    std_errors2 = np.zeros(n)
    for i in Ks:
        errors2[i-1], std_errors2[i-1] = test_random_projection_with_kmeans2(n,i)

    errors2[errors2==0] = 1
    std_errors2[0] = 0
    print errors
    print errors2
    plt.figure()
    plt.errorbar(Ks,errors,std_errors)
    plt.errorbar(np.arange(1,n+1),errors2,std_errors2)
    plt.plot(Ks,np.sqrt(Ks-Ks**2/n))


    plt.figure()
    meas_errors = np.array([0] + [e for e in errors[::-1]])
    meas_errors[meas_errors==0] = 1
    plt.plot(np.arange(1,n+1),(meas_errors-errors2)/meas_errors)


def identify_next_level(A,Ks, model='SBM',reg=False, norm='F', threshold=1/3, reps=20, noise=1e-3):
    """
    Identify agglomeration levels by checking the projection errors and comparing the to a bootstrap
    verstion of the same network"""

    #first identify partitions and their projection error
    Ks, sum_errors, partition_vecs = identify_partitions_and_errors(A,Ks,model,reg, norm,partition_vecs=[])

    #repeat with noise
    if reps>0:

        sum_errors = 0.
        std_errors = 0.
        m = 0.

        for rep in xrange(reps):
            Anew = spectral.add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(Anew,Ks,model,reg, norm,partition_vecs)
            sum_errors+=errors

            #calculate online variance
            m_prev = m
            m = m + (errors - m) / (reps+1)
            std_errors = std_errors + (errors - m) * (errors - m_prev)


        sum_errors/=reps
    std_errors=np.sqrt(std_errors)

    return sum_errors, std_errors



def identify_partitions_and_errors(A,Ks,model='SBM',reg=False, norm='F',partition_vecs=[]):
    """
    Collect the partitions and projection errors found for a list Ks of 'putative' group numbers
    """

    max_k = Ks[0]

    L = spectral.construct_graph_Laplacian(A)
    Dtau_sqrt_inv = 0
    tau = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        print L.shape, max_k
        ev, evecs = scipy.sparse.linalg.eigsh(L,Ks[0],which='SM',tol=1e-6)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index]

    #initialise errors
    error = np.zeros(len(Ks))
    #check if partitions are known
    partitions_unknown= partition_vecs==[]

    #find partitions and their error for each k
    for ki,k in enumerate(Ks):
        if partitions_unknown:
            partition_vec, Hnorm = spectral.find_partition(evecs, k, tau, norm, model, Dtau_sqrt_inv,method='QR')
        else :
            partition_vec = partition_vecs[ki]
            Hnorm = spectral.create_normed_partition_matrix_from_vector(partition_vec,model)

        #calculate and store error
        error[ki] = calculate_proj_error3(evecs, Hnorm, norm)
        # print("K, error ")
        # print(k, error[ki])

        #save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs
