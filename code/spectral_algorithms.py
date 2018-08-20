#!/usr/bin/env python
from __future__ import division

import networkx as nx
from matplotlib import pyplot as plt
from sys import stdout

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import LinearOperator
from scipy.signal import argrelextrema
import scipy.linalg
import scipy.stats

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn import mixture

#TODO: we probably want to have a more clean import.
from helperfunctions import *

def hier_spectral_partition(A,spectral_oper='Lap',first_pass='Bethe', model='SBM', reps=10, noise=1e-3, Ks=None):
    """
    Performs a full round of hierarchical spectral clustering using the configurations provided.
    Ks is a list of the partition sizes at each level in inverse order (e.g. [27, 9, 3] for a hierarchical
    split into 3 x 3 x 3 groups.
    reps defines the number of sampples drawn in the bootstrap error test and noise the corresponding noise floor.
    """

    # if Ks is not specified then perform complete infernce
    if Ks is None:
        # initial spectral clustering, performed according to method 'first pass'
        p0 = spectral_partition(A,spectral_oper=first_pass,num_groups=-1)
        Ks=[]
        Ks.append(np.max(p0)+1)

    #otherwise Ks specifies partition sizes for each level (or just finest level if length 1) from finest to coarsest
    #~ else:

    #TODO: atm the above just uses the Bethe Hessian to figure out the number of groups but then clusters using the Rohe Laplacian
    p0 = spectral_partition(A,spectral_oper=spectral_oper,num_groups=Ks[0])

    # if Ks only specifies starting partition then set to none
    if len(Ks)==1:
        Ks=None
    else:
        Ks=Ks[1:]

    # agglomerate builds list of all partitions
    pvec_agg = hier_spectral_partition_agglomerate(A,p0, spectral_oper=spectral_oper, model=model, reps=reps, noise=noise, Ks=Ks)


    return pvec_agg


def hier_spectral_partition_agglomerate(A, partition, spectral_oper="Lap", model='SBM', reps=10, noise=1e-3, Ks=None):
    """Performs hierarchical agglomeration of adjacency matrix and provided partition,
        based on the provided mode parameter"""

    # pvec stores all hier. refined partitions
    pvec = []
    pvec.append(partition)

    # k == index of highest groups / number of groups = k+1
    k = np.max(partition)

    print "HIER SPECTRAL PARTITION -- agglomerative\n Initial partition into", k+1, "groups \n"

   # Ks stores the candidate levels in inverse order
   # Note: set to min 1 group, as no agglomeration required when only 2 groups are detected.
    if Ks is None:
        Ks=np.arange(k,1,-1)
        find_levels = True
    else:
        find_levels = False

    # levels is a list of 'k' values of each level in the inferred hierarchy
    levels = [k+1]
    while len(Ks)>0:

        Eagg, Nagg = compute_number_links_between_groups(A,partition)
        # TODO The normalization seems important for good results -- why?
        Aagg = Eagg / Nagg

        if find_levels:
            Ks, hier_partition_vecs = identify_next_level(Aagg,Ks,model=model,reg=False, norm='F', threshold=1/3, reps=reps, noise=noise)
        else:
            k = Ks[0]-1
            Ks, hier_partition_vecs = identify_partitions_at_level(Aagg,Ks,model=model,reg=False, norm='F')

        try:
            pvec.append(hier_partition_vecs[0])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            if find_levels:
                k=Ks[0]-1
                Ks = np.arange(k,1,-1)
            levels.append(k+1)
            #if levels are not prespecified, reset candidates
            print 'partition into', k+1 ,' groups'
            if k==1:
                Ks=[]

        #TODO: check if below is not better described as: if there is *no candidate* partition (why only single candidate? why error not high enough -- low?!)
        #this exception occurs when there is only a single candidate partition and the error is not high enough.
        except IndexError:
            pass


    print "HIER SPECTRAL PARTITION -- agglomerative\n Partitions into", levels, "groups \n"

    return expand_partitions_to_full_graph(pvec)[::-1]

def spectral_partition(A, spectral_oper='Lap', num_groups=2, regularizer='BHa'):
    """ Perform one round of spectral clustering for a given network matrix A
    Inputs: A -- input adjacency matrix
            mode -- variant of spectral clustering to use (Laplacian, Bethe Hessian,
            Non-Backtracking, XLaplacian, ...)
            num_groups -- in how many groups do we want to split the graph?
            (default: 2; set to -1 to infer number of groups from spectrum)

            Output: partition_vec -- clustering of the nodes
    """

    if   spectral_oper == "Lap":
        if num_groups != -1:
            partition, _ = regularized_laplacian_spectral_clustering(A,num_groups=num_groups)

    elif spectral_oper == "Bethe":
        partition = cluster_with_BetheHessian(A,num_groups=num_groups,mode='unweighted', regularizer=regularizer)

    else:
        raise ValueError("mode '%s' not recognised - available modes are 'Lap', Bethe'" % mode)

    partition = relabel_partition_vec(partition)
    return partition


##########################################
# REGULARIZED SPECTRAL CLUSTERING (ROHE)
##########################################

def regularized_laplacian_spectral_clustering(A, num_groups=2, tau=-1,clustermode='kmeans'):
    """
    Performs regularized spectral clustering based on Qin-Rohe 2013 using a normalized and
    regularized adjacency matrix (called Laplacian by Rohe et al)
    """

    A = test_sparse_and_transform(A)

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau==-1:
        # set tau to average degree
        tau = A.sum()/A.shape[0]

    L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A,tau)


    # compute eigenvalues and eigenvectors (sorted according to magnitude first)
    ev, evecs = scipy.sparse.linalg.eigsh(L,num_groups,which='LM',tol=1e-6)

    if clustermode == 'kmeans':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')

        clust = KMeans(n_clusters = num_groups)
        clust.fit(X)
        partition_vector = clust.labels_
    elif clustermode == 'qr':
        partition_vector = clusterEVwithQR(evecs)


    return partition_vector, evecs

def construct_normalised_Laplacian(Omega, reg):
    """
    Construct a normalized regularized Laplacian matrix from the input matrix Omega
    Input reg can be bool in which case the default settings for regularization (yes/no)
    are used or a float in which case the numerical values is used as a regularizer
    """
    if isinstance(reg,bool):
        if reg:
            # set tau to average degree
            tau = Omega.sum()/Omega.shape[0]
        else:
            tau = 0
    else:
        tau =reg

    # construct normalised Laplacian either as sparse matrix or dense array depending on input
    if scipy.sparse.issparse(Omega):
        Dtau_sqrt_inv = scipy.sparse.diags(np.power(np.array(Omega.sum(1)).flatten().astype(float) + tau,-.5),0)
        L = Dtau_sqrt_inv.dot(Omega).dot(Dtau_sqrt_inv)
    else:
        Dtau_sqrt_inv = scipy.diagflat(np.power(np.array(Omega.sum(1)).flatten().astype(float) + tau,-.5),0)
        L = Dtau_sqrt_inv.dot(Omega).dot(Dtau_sqrt_inv)

    return L, Dtau_sqrt_inv, tau

def construct_graph_Laplacian(Omega):
    """
    Construct a Laplacian matrix from the input matrix Omega. Output can be a sparse
    matrix or a dense array depending on input
    """
    # construct Laplacian either as sparse matrix or dense array depending on input
    if scipy.sparse.issparse(Omega):
        D = scipy.sparse.diags(Omega.sum(1).flatten().astype(float),0)
        L = D - Omega
    else:
        D = scipy.diagflat(Omega.sum(1).flatten().astype(float),0)
        L = D - Omega

    return L

######################################
# BETHE HESSIAN CLUSTERING
######################################

def build_BetheHessian(A, r):
    """
    Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
    B = (r^2-1)*I-r*A+D
    """
    A = test_sparse_and_transform(A)

    d = A.sum(axis=1).getA().flatten().astype(float)
    B = scipy.sparse.eye(A.shape[0]).dot(r**2 -1) -r*A +  scipy.sparse.diags(d,0)
    return B


def build_weighted_BetheHessian(A,r):
    """
    Construct weigthed Bethe Hessian as discussed in Saade et al.
    """
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)

    # we are interested in A^.2 (elementwise)
    A2data = A.data **2

    new_data = A2data / (r*r -A2data)
    A2 = scipy.sparse.csr_matrix((new_data,A.nonzero()),shape=A.shape)

    # diagonal matrix
    d = 1 + A2.sum(axis=1)
    d = d.getA().flatten()
    DD = scipy.sparse.diags(d,0)

    # second matrix
    rA_data = r*A.data / (r*r - A2data)
    rA = scipy.sparse.csr_matrix((rA_data,A.nonzero()),shape=A.shape)

    # full Bethe Hessian
    BHw = DD - rA
    return BHw


def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa',
                              mode='weighted',clustermode='kmeans'):
    """
    Perform one round of spectral clustering using the Bethe Hessian
    """

    if regularizer=='BHa':
        # set r to square root of average degree
        r = A.sum()/A.shape[0]
        r = np.sqrt(r)

    elif regularizer=='BHm':
        d = A.sum(axis=1).getA().flatten().astype(float)
        r = np.sum(d*d)/np.sum(d) - 1
        r = np.sqrt(r)

    if all(A.sum(axis=1) == 0):
        # print "empty Graph -- return all in one partition"
        partition_vector = np.zeros(A.shape[0],dtype='int')
        return partition_vector

    # construct both the positive and the negative variant of the BH
    if mode == 'unweighted':
        BH_pos = build_BetheHessian(A,r)
        BH_neg = build_BetheHessian(A,-r)
    elif mode == 'weighted':
        BH_pos = build_weighted_BetheHessian(A,r)
        BH_neg = build_weighted_BetheHessian(A,-r)
    else:
        print "Something went wrong"
        return -1


    if num_groups ==-1:
        relevant_ev, lambda1 = find_negative_eigenvectors(BH_pos)
        X = relevant_ev

        relevant_ev, lambda2 = find_negative_eigenvectors(BH_neg)
        X = np.hstack([X, relevant_ev])
        print "number nodes /groups"
        print X.shape
        # print "Xvectors"
        # print X
        num_groups = X.shape[1]
        num_samples = X.shape[0]

        if num_groups == 0 or num_samples < num_groups:
            print "no indication for grouping -- return all in one partition"
            partition_vector = np.zeros(A.shape[0],dtype='int')
            return partition_vector

    else:
        # TODO: note that we combine the eigenvectors of pos/negative BH and do not use
        # information about positive / negative assortativity here
        # find eigenvectors corresponding to the algebraically smallest (most neg.) eigenvalues
        ev_pos, evecs_pos = scipy.sparse.linalg.eigsh(BH_pos,num_groups,which='SA',tol=1e-6)
        ev_neg, evecs_neg = scipy.sparse.linalg.eigsh(BH_neg,num_groups,which='SA',tol=1e-6)
        ev_all = np.hstack([ev_pos, ev_neg])
        index = np.argsort(ev_all)
        X = np.hstack([evecs_pos,evecs_neg])
        X = X[:,index[:num_groups]]


    if clustermode == 'kmeans':
        clust = KMeans(n_clusters = num_groups)
        clust.fit(X)
        partition_vector = clust.labels_
    elif clustermode == 'qr':
        partition_vector = clusterEVwithQR(X)
    else:
        print "Something went wrong -- provide valid clustermode"


    return partition_vector

##################################################
# SPECTRAL MODEL SELECTION VIA INVARIANT SUBSPACE
##################################################

def identify_partitions_at_level(A,Ks,model='SBM',reg=False, norm='F'):
    """
    For a given graph with (weighted) adjacency matrix A and list of partition sizes to assess (Ks),
    find the partition of a given size Ks[0] via the find_partition function using the model and regularization
    provided.
    """

    # L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A, reg)
    #TODO: check here
    L = construct_graph_Laplacian(A)
    Dtau_sqrt_inv = 0
    tau = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    sparse_input = False
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        # ev, evecs = scipy.sparse.linalg.eigsh(L,Ks[0],which='LM',tol=1e-6)
        #TODO: check here
        ev, evecs = scipy.sparse.linalg.eigsh(L,Ks[0],which='SM',tol=1e-6)
        sparse_input = True

    index = np.argsort(np.abs(ev))
    #TODO: check here
    # evecs = evecs[:,index[::-1]]
    evecs = evecs[:,index]

    # find the partition for Ks[0] groups
    partition_vec, Hnorm = find_partition(evecs, Ks[0], tau, norm, model, Dtau_sqrt_inv)

    return Ks[1:], [partition_vec]


def identify_next_level(A,Ks, model='SBM',reg=False, norm='F', threshold=1/3, reps=10, noise=1e-3):
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
            Anew = add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(Anew,Ks,model,reg, norm,partition_vecs)
            sum_errors+=errors

            #calculate online variance
            m_prev = m
            m = m + (errors - m) / (reps+1)
            std_errors = std_errors + (errors - m) * (errors - m_prev)


        sum_errors/=reps

    std_errors=np.sqrt(std_errors)
    #find errors below threshold
    below_thresh = (sum_errors<threshold)


    #TODO: replace all of the below and the find minima function by using find peaks function!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    levels = find_local_minima(sum_errors)
    print below_thresh.nonzero()[0]
    print 'sum_errors',zip(Ks[levels],sum_errors[levels])
    print 'sum_errors',zip(Ks[below_thresh],sum_errors[below_thresh])
    #choose only local minima that are below threshold
    levels = np.intersect1d(levels,below_thresh.nonzero()[0])
    print 'Levels inferred=',len(levels), Ks[levels], sum_errors[levels], std_errors[levels]
    hier_partition_vecs=[partition_vecs[si] for si in levels]
    return Ks[levels], hier_partition_vecs


def identify_partitions_and_errors(A,Ks,model='SBM',reg=False, norm='F',partition_vecs=[]):
    """
    Collect the partitions and projection errors found for a list Ks of 'putative' group numbers
    """

    max_k = Ks[0]

    #TODO: check here!
    # L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A, reg)
    L = construct_graph_Laplacian(A)
    Dtau_sqrt_inv = L
    tau = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    sparse_input = False
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        print L.shape, max_k
        # ev, evecs = scipy.sparse.linalg.eigsh(L,max_k,which='LM',tol=1e-6)
        #TODO: check here
        ev, evecs = scipy.sparse.linalg.eigsh(L,max_k,which='SM',tol=1e-6)
        sparse_input = True
    index = np.argsort(np.abs(ev))
    #TODO: check here
    # evecs = evecs[:,index[::-1]]
    evecs = evecs[:,index]

    #initialise errors
    error = np.zeros(len(Ks))
    #check if partitions are known
    partitions_unknown= partition_vecs==[]

    #find partitions and their error for each k
    for ki,k in enumerate(Ks):
        if partitions_unknown:
            partition_vec, Hnorm = find_partition(evecs, k, tau, norm, model, Dtau_sqrt_inv)
        else :
            partition_vec = partition_vecs[ki]
            Hnorm = create_normed_partition_matrix_from_vector(partition_vec,model)

        #calculate and store error
        error[ki] = calculate_proj_error(evecs, Hnorm, norm)
        #~ print("K, error ")
        #~ print(k, error[ki])

        #save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs


def find_partition(evecs, k, tau, norm, model, Dtau_sqrt_inv, method='QR', n_init=20):
    """ Perform clustering in spectral embedding space according to various criteria"""
    V = evecs[:,:k]

    if model == 'DCSBM':
        X = preprocessing.normalize(V, axis=1, norm='l2')
    elif model == 'SBM':
        # X = Dtau_sqrt_inv* V
        X = V
    else:
        error('something went wrong. Please specify valid mode')

    #select methof of clustering - QR or KM (k-means)
    if method=='QR':
        partition_vec = clusterEVwithQR(X)
    elif method=='KM':
        clust = KMeans(n_clusters = k, n_init=n_init)
        clust.fit(X)
        partition_vec = clust.labels_
        partition_vec = relabel_partition_vec(partition_vec)
    else:
        error('something went wrong. Please specify valid clustering method')

    H = create_normed_partition_matrix_from_vector(partition_vec,model)

    return partition_vec, H

def calculate_proj_error(evecs,H,norm):
    """ Given a set of eigenvectors and a partition matrix, try project compute the alignment between those two subpacees by computing the projection (errors) of one into the other"""
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

def project_orthogonal_to(subspace_basis,vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space orthogonal to which we want to project
    vectors_to_project: project these vectors into the orthogonal complement of the
    specified subspace

    compute S*(S^T*S)^{-1}*S' * V
    """

    if not scipy.sparse.issparse(vectors_to_project):
        V = np.matrix(vectors_to_project)
    else:
        V = vectors_to_project

    if not scipy.sparse.issparse(subspace_basis):
        S = np.matrix(subspace_basis)
    else:
        S = subspace_basis

    projected = S*scipy.sparse.linalg.spsolve(S.T*S,S.T*V)

    orthogonal_proj = V - projected
    return orthogonal_proj

##################################################
# QR Decomposition for finding clusters
##################################################

def clusterEVwithQR(EV, randomized=False, gamma=4):
    """Given a set of eigenvectors find the clusters of the SBM"""
    if randomized is True:
        Z, Q = orthogonalizeQR_randomized(EV,gamma)
    else:
        Z, Q = orthogonalizeQR(EV)

    cluster_ = scipy.absolute(Z).argmax(axis=1).astype(int)

    return cluster_

def orthogonalizeQR(EV):
    """Given a set of eigenvectors V coming from a operator associated to the SBM,
    use QR decomposition as described in Damle et al 2017, to compute new coordinate
    vectors aligned with clustering vectors
    Input EV is an N x k matrix where each column corresponds to an eigenvector
    """
    k = EV.shape[1]
    Q, R, P = scipy.linalg.qr(EV.T, mode='economic', pivoting=True)
    # get indices of k representative points
    P = P[:k]

    # polar decomposition to find nearest orthogonal matrix
    U, S, V = scipy.linalg.svd(EV[P,:].T,full_matrices=False)

    #TODO: check this part!
    # Z = EV.dot(U.dot(V.T))
    Z = EV.dot(EV[P,:].T)

    return Z, Q

def orthogonalizeQR_randomized(EV,gamma=4):
    """Given a set of eigenvectors V coming from a operator associated to the SBM,
    use randomized QR decomposition as described in Damle et al 2017, to compute new
    coordinate vectors aligned with clustering vectors.

    Input EV is an N x k matrix where each column corresponds to an eigenvector
    gamma is the oversampling factor
    """
    n, k = EV.shape

    # sample EV according to leverage scores and the build QR from those vectors
    count = scipy.minimum(scipy.ceil(gamma*k*scipy.log(k)),n)
    elements = np.arange(n)
    prob = (EV.T**2).sum(axis=0)
    probabilities = prob / prob.sum()
    elements = np.random.choice(elements, count, p=probabilities)
    elements = scipy.unique(elements)


    Q, R, P = scipy.linalg.qr(EV[elements,:].T, mode='economic', pivoting=True)
    # get indices of k representative points
    P = P[:k]

    # polar decomposition to find nearest orthogonal matrix
    U, S, V = scipy.linalg.svd(EV[P,:].T,full_matrices=False)

    Z = EV.dot(U.dot(V.T))

    return Z, Q

