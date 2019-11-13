#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:
1) Functions to construct various spectral operators
2) Functions for spectral clustering based on eigendecompositions of these operators
3) Functions for checking hier. agglomerations based on the obtained partitions

"""
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

########################################
# PART 1 -- construct spectral operators
########################################

def construct_normalised_Laplacian(Omega, reg):
    """
    Construct a normalized regularized Laplacian matrix (normalized adjaceny)
    Input:
        Omega -- Matrix / Array of network
        reg  -- if reg is of type bool (true / false) then it acts as a switch for
                regularization according to average degree;
                otherwise acts as a numerical value for the regularizer

    are used or a float in which case the numerical values is used as a regularizer
    """
    if isinstance(reg, bool):
        if reg:
            # set tau to average degree
            tau = Omega.sum() / Omega.shape[0]
        else:
            tau = 0
    else:
        tau = reg

    degree = np.array(Omega.sum(1)).flatten().astype(float) + tau

    # construct normalised Laplacian either as sparse matrix or dense array depending on input
    if scipy.sparse.issparse(Omega):
        Dtau_sqrt_inv = scipy.sparse.diags(np.power(degree, -.5), 0)
        L = Dtau_sqrt_inv.dot(Omega).dot(Dtau_sqrt_inv)
    else:
        Dtau_sqrt_inv = scipy.diagflat(np.power(degree, -.5), 0)
        L = Dtau_sqrt_inv.dot(Omega).dot(Dtau_sqrt_inv)

    return L, Dtau_sqrt_inv, tau


def construct_graph_Laplacian(Omega):
    """
    Construct a Laplacian matrix from the input matrix Omega. Output can be a sparse
    matrix or a dense array depending on input
    """
    # construct Laplacian either as sparse matrix or dense array depending on input
    if scipy.sparse.issparse(Omega):
        D = scipy.sparse.diags(Omega.sum(1).flatten().astype(float), 0)
        L = D - Omega
    else:
        D = scipy.diagflat(Omega.sum(1).flatten().astype(float), 0)
        L = D - Omega

    return L


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

############################################
# PART 2 -- single pass spectral clustering
############################################

def spectral_partition(A, spectral_oper='Lap', num_groups=2, regularizer='BHa'):
    """ Perform one round of spectral clustering for a given network matrix A
    Inputs: A -- input adjacency matrix
            mode -- variant of spectral clustering to use (Laplacian, Bethe Hessian,
            Non-Backtracking, XLaplacian, ...)
            num_groups -- in how many groups do we want to split the graph?
            (default: 2; set to -1 to infer number of groups from spectrum)

            Output: partition_vec -- clustering of the nodes
    """

    if spectral_oper == "Lap":
        if num_groups != -1:
            partition, _ = regularized_laplacian_spectral_clustering(A, num_groups=num_groups)

    elif spectral_oper == "Bethe":
        partition = cluster_with_BetheHessian(A, num_groups=num_groups, mode='unweighted', regularizer=regularizer)

    else:
        raise ValueError("mode '%s' not recognised - available modes are 'Lap', Bethe'" % spectral_oper)

    partition = relabel_partition_vec(partition)
    return partition


def regularized_laplacian_spectral_clustering(A, num_groups=2, tau=-1, clustermode='GMM', n_init=10):
    """
    Performs regularized spectral clustering based on Qin-Rohe 2013 using a normalized and
    regularized adjacency matrix (called Laplacian by Rohe et al)
    """

    A = test_sparse_and_transform(A)

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau == -1:
        # set tau to average degree
        tau = A.sum() / A.shape[0]

    laplacian, _, tau = construct_normalised_Laplacian(A, tau)

    # compute eigenvalues and eigenvectors (sorted according to magnitude first)
    ev, evecs = scipy.sparse.linalg.eigsh(laplacian, num_groups, which='LM', tol=1e-6)

    if clustermode == 'kmeans':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')

        clust = KMeans(n_clusters=num_groups, n_init=n_init)
        clust.fit(X)
        partition_vector = clust.labels_

    elif clustermode == 'qr':
        partition_vector = clusterEVwithQR(evecs)

    elif clustermode == 'GMM':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')
        GMM = mixture.GaussianMixture(n_components=num_groups, covariance_type='full', n_init=n_init)
        GMM.fit(X)
        partition_vector = GMM.predict(X)

    elif clustermode == 'GMMdiag':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')
        GMM = mixture.GaussianMixture(n_components=num_groups, covariance_type='diag', n_init=n_init)
        GMM.fit(X)
        partition_vector = GMM.predict(X)

    elif clustermode == 'GMMsphere':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')
        GMM = mixture.GaussianMixture(n_components=num_groups, covariance_type='sphere', n_init=n_init)
        GMM.fit(X)
        partition_vector = GMM.predict(X)

    partition_vector = relabel_partition_vec(partition_vector)

    return partition_vector, evecs


def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa', mode='weighted', clustermode='kmeans'):
    """
    Perform one round of spectral clustering using the Bethe Hessian
    """

    if regularizer == 'BHa':
        # set r to square root of average degree
        r = A.sum() / A.shape[0]
        r = np.sqrt(r)

    elif regularizer == 'BHm':
        d = A.sum(axis=1).getA().flatten().astype(float)
        r = np.sum(d * d) / np.sum(d) - 1
        r = np.sqrt(r)

    if all(A.sum(axis=1) == 0):
        # print "empty Graph -- return all in one partition"
        partition_vector = np.zeros(A.shape[0], dtype='int')
        return partition_vector

    # construct both the positive and the negative variant of the BH
    if mode == 'unweighted':
        BH_pos = build_BetheHessian(A, r)
        BH_neg = build_BetheHessian(A, -r)
    elif mode == 'weighted':
        BH_pos = build_weighted_BetheHessian(A, r)
        BH_neg = build_weighted_BetheHessian(A, -r)
    else:
        print "Something went wrong"
        return -1

    if num_groups == -1:
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
            partition_vector = np.zeros(A.shape[0], dtype='int')
            return partition_vector

    else:
        # TODO: note that we combine the eigenvectors of pos/negative BH and do not use
        # information about positive / negative assortativity here
        # find eigenvectors corresponding to the algebraically smallest (most neg.) eigenvalues
        ev_pos, evecs_pos = scipy.sparse.linalg.eigsh(BH_pos, num_groups, which='SA', tol=1e-6)
        ev_neg, evecs_neg = scipy.sparse.linalg.eigsh(BH_neg, num_groups, which='SA', tol=1e-6)
        ev_all = np.hstack([ev_pos, ev_neg])
        index = np.argsort(ev_all)
        X = np.hstack([evecs_pos, evecs_neg])
        X = X[:, index[:num_groups]]

    if clustermode == 'kmeans':
        clust = KMeans(n_clusters=num_groups)
        clust.fit(X)
        partition_vector = clust.labels_
    elif clustermode == 'qr':
        partition_vector = clusterEVwithQR(X)
    else:
        print "Something went wrong -- provide valid clustermode"

    return partition_vector


def clusterEVwithQR(EV, randomized=False, gamma=4):
    """Given a set of eigenvectors find the clusters of the SBM"""
    if randomized is True:
        Z, Q = orthogonalizeQR_randomized(EV, gamma)
    else:
        Z, Q = orthogonalizeQR(EV)

    cluster_ = scipy.absolute(Z).argmax(axis=1).astype(int)

    return cluster_


##############################################################
# PART 3 -- Hierarchical spectral clustering and agglomeration
##############################################################

def hier_spectral_partition(A, spectral_oper='Lap', first_pass='Bethe', model='SBM', reps=10, noise=1e-2, Ks=None):
    """
    Performs a full round of hierarchical spectral clustering.
    Inputs:
        A -- sparse {0,1} adjacency matrix of network
        first_pass -- spectral operator for the first pass partitioning
        spectral_oper -- spectral operator for remaining passes over data
        model -- parameter for spectral clustering
        reps -- repetitions / number of samples drawn in bootstrap error test
        noise -- corresponding noise parameter for bootstrap
        Ks -- list of the partition sizes at each level in inverse order
              (e.g. [27, 9, 3] for a hierarchical split into 3 x 3 x 3 groups.

    Output: list of hier. partitions

    The algorithm proceeds by
    1) estimate the initial number of groups if Ks is not provided
    2) get initial partition of the network using a method for clustering sparse networks
    3) find partitions which fulfill the EEP criterion and agglomerate
    """

    # FIRST STEP -- estimate number of partitions
    # If Ks is not specified then estimate the number of clusters using the Bethe Hessian
    # TODO: atm just uses the Bethe Hessian to figure out the number of communities but
    # then uses Rohe Laplacian to do the first inference step -- we can simplify to just
    # use Bethe Hessian
    if Ks is None:
        p0 = spectral_partition(A, spectral_oper=first_pass, num_groups=-1)
        Ks = []
        Ks.append(np.max(p0) + 1)

    # SECOND STEP
    # initial spectral clustering using spectral_oper; Lap == Rohe Laplacian
    p0 = spectral_partition(A, spectral_oper=spectral_oper, num_groups=Ks[0])

    # if Ks only specifies number of clusters at finest level,
    # set again to none for agglomeration
    if len(Ks) == 1:
        Ks = None
    else:
        Ks = Ks[1:]

    # THIRD STEP
    # Now we build a list of all partitions
    pvec_agg = hier_spectral_partition_agglomerate(
        A, p0, spectral_oper=spectral_oper, model=model, reps=reps, noise=noise, Ks=Ks)

    return pvec_agg


def hier_spectral_partition_agglomerate(A, partition, spectral_oper="Lap",
                                        model='SBM', reps=10, noise=1e-2, Ks=None, no_Ks_forward=True):
    """
    Given a graph A and an initial partition, check for possible agglomerations within
    the network.
    Inputs:
        A -- network as sparse adjacency matrix
        partition -- initial partition
        reps -- repetitions / number of samples drawn in bootstrap error test
        noise -- corresponding noise parameter for bootstrap
        Ks -- list of the partition sizes at each level in inverse order
              (e.g. [27, 9, 3] for a hierarchical split into 3 x 3 x 3 groups.
        spectral_oper -- spectral operator used to perform clustering not used atm
        model -- parameter for spectral clustering

    Outputs:
        List of hier. partitions


    The agglomerative clustering proceeds as follows:
    1) If the list of Ks to consider is empty, we will consider all possible Ks and need
    to find the right subsplits/ levels, otherwise we only need to check partitions for
    the number of groups provided
    2) While the list of of Ks is not empty, agglomerate network and try to identify
    partitions either using the identify_next_level function
    (unknown Ks, full list), or the identify_partitions_at_level function
    (known Ks, list of group sizes)
    3) these functions return a new Ks list, and a new partition to operate with
    according to which we agglomerate the network and start again from 1)
    """
    # TODO: the following options are not really made use of
    # spectral_oper -- spectral operator used to perform clustering not used atm
    # model -- parameter for spectral clustering

    # k == index of highest groups / number of groups = k+1
    k = np.max(partition)
    print "HIER SPECTRAL PARTITION -- agglomerative\n"
    print "Initial partition into", k + 1, "groups \n"

    # Ks stores the candidate levels in inverse order
    # Note: set to min 1 group, as no agglomeration required
    # when only 2 groups are detected.
    if Ks is None:
        Ks = np.arange(k, 1, -1)
        find_levels = True
    else:
        find_levels = False

    # levels is a list of 'k' values of each level in the inferred hierarchy
    # pvec stores all hier. refined partitions
    levels = [k + 1]
    pvec = []
    pvec.append(partition)
    while len(Ks) > 0:

        print "List of partitions to assess: ", Ks, "\n"
        print "Current shape of network: ", A.shape, "\n"

        # Question: should we normalize by varaiance of Bernoullis as well?
        Eagg, Nagg = compute_number_links_between_groups(A, partition)
        Aagg = Eagg / Nagg

        if find_levels:
            Ks, hier_partition_vecs = identify_next_level(
                Aagg, Ks, model=model, reg=False, norm='F', threshold=1 / 3, reps=reps, noise=noise)
        else:
            k = Ks[0] - 1
            Ks, hier_partition_vecs = identify_partitions_at_level(
                Aagg, Ks, model=model, reg=False, norm='F')

        try:
            pvec.append(hier_partition_vecs[0])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            if find_levels:
                if no_Ks_forward:
                    k = Ks[0] - 1
                    Ks = np.arange(k, 1, -1)
                else:
                    # It might be useful to pass down candidates
                    # from previous agglomeration rounds here instead of starting from scratch?!
                    # TODO implement this!?
                    pass
            levels.append(k + 1)
            # if levels are not prespecified, reset candidates
            print 'partition into', k + 1, ' groups'
            if k == 1:
                Ks = []

        # this exception occurs when there is only a single candidate partition
        # and the error is not high enough
        # Note: is this not better described as: if there is *no candidate* partition
        # (why only single candidate? why error not high enough -- low?!)
        except IndexError:
            pass

    print "HIER SPECTRAL PARTITION -- agglomerative\n Partitions into", levels, "groups \n"

    return expand_partitions_to_full_graph(pvec)[::-1]

# -------------------------------------------------
# SPECTRAL MODEL SELECTION VIA INVARIANT SUBSPACES
# -------------------------------------------------


def identify_partitions_at_level(A, Ks, model='SBM', reg=False, norm='F'):
    """
    For a given graph with (weighted) adjacency matrix A and list of
    partition sizes to assess (Ks), find the partition of a given size Ks[0]
    via the find_partition function using the model and regularization
    provided.
    Inputs:
        A -- (agglomerated) adjacency matrix of network
        Ks -- list of the partition sizes at each level in inverse order
              (e.g. [27, 9, 3] for a hierarchical split into 3 x 3 x 3 groups.
        model -- parameter for spectral clustering
        reg -- use regularization for spectral clustering?
        norm -- unclear

    Outputs:
        Ks -- remaining list of group sizes to consider
        partition_vec -- found partition at this level
    """
    #TODO: norm should not be passed on as parameter here, since we do not assess error.

    # L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A, reg)
    L = construct_graph_Laplacian(A)
    Dtau_sqrt_inv = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        # ev, evecs = scipy.sparse.linalg.eigsh(L,Ks[0],which='LM',tol=1e-6)
        ev, evecs = scipy.sparse.linalg.eigsh(L, Ks[0], which='SM', tol=1e-6)

    index = np.argsort(np.abs(ev))
    # evecs = evecs[:,index[::-1]]
    evecs = evecs[:, index]

    # find the partition for Ks[0] groups
    partition_vec, Hnorm = find_partition(evecs, Ks[0], model, Dtau_sqrt_inv)

    return Ks[1:], [partition_vec]


def identify_next_level(A, Ks, model='SBM', reg=False, norm='F', threshold=1 / 3, reps=10, noise=1e-2):
    """
    Identify agglomeration levels by checking the projection errors and comparing the to a
    perturbed verstion of the same network
    Inputs:
        A -- (agglomerated) adjacency matrix of network
        Ks -- list of the partition sizes at each level in inverse order
              (e.g. [27, 9, 3] for a hierarchical split into 3 x 3 x 3 groups.
        model -- parameter for spectral clustering
        reg -- use regularization for spectral clustering?
        norm -- norm used to assess projection error
        threshold -- threshold parameter to assess projection error
        reps -- number of repetitions for bootstrap
        noise -- noise parameter for bootstrap

    Outputs:
        Ks -- remaining list of group sizes to consider
        hier_partition_vecs -- found putative hier partitions

    """

    # first identify partitions and their projection error
    Ks, sum_errors, partition_vecs = identify_partitions_and_errors(
        A, Ks, model, reg, norm, partition_vecs=[])

    # repeat with noise
    if reps > 0:
        sum_errors = 0.
        std_errors = 0.
        m = 0.

        for rep in xrange(reps):
            Anew = add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(
                Anew, Ks, model, reg, norm, partition_vecs)
            sum_errors += errors

            # calculate online variance
            m_prev = m
            m = m + (errors - m) / (rep + 1)
            std_errors = std_errors + (errors - m) * (errors - m_prev)

        sum_errors /= reps
        std_errors = np.sqrt(std_errors/(reps-1))
    # find errors below threshold
    below_thresh = (sum_errors < threshold)

    # TODO: replace all of the below and the find minima function by using find peaks function!?
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    levels = find_local_minima(sum_errors)
    # print below_thresh.nonzero()[0]
    # print 'sum_errors', zip(Ks[levels], sum_errors[levels])
    # print 'sum_errors', zip(Ks[below_thresh], sum_errors[below_thresh])
    # choose only local minima that are below threshold
    levels = np.intersect1d(levels, below_thresh.nonzero()[0])
    print 'Levels inferred=', len(levels), Ks[levels], sum_errors[levels], std_errors[levels]
    hier_partition_vecs = [partition_vecs[si] for si in levels]
    return Ks[levels], hier_partition_vecs


def identify_partitions_and_errors(A, Ks, model='SBM', reg=False, norm='F', partition_vecs=[]):
    """
    Collect the partitions and projection errors found for a list Ks of
    'putative' group numbers

    Inputs:
        A -- (agglomerated) adjacency matrix of network
        Ks -- list of the partition sizes at each level in inverse order
              (e.g. [27, 9, 3] for a hierarchical split into 3 x 3 x 3 groups.
        model -- parameter for spectral clustering
        reg -- use regularization for spectral clustering?
        norm -- norm used to assess projection error
        partition_vecs -- list of partitions from previous clustering, whos error w.r.t.
        the eigenvectors of the new network we would like to assess.
    """

    max_k = Ks[0]

    # L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A, reg)
    L = construct_graph_Laplacian(A)
    Dtau_sqrt_inv = L

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        print L.shape, max_k
        # ev, evecs = scipy.sparse.linalg.eigsh(L,max_k,which='LM',tol=1e-6)
        ev, evecs = scipy.sparse.linalg.eigsh(L, max_k, which='SM', tol=1e-6)
    index = np.argsort(np.abs(ev))
    # evecs = evecs[:,index[::-1]]
    evecs = evecs[:, index]

    # initialise errors
    error = np.zeros(len(Ks))
    # check if partitions are known
    partitions_unknown = partition_vecs == []

    # find partitions and their error for each k
    for ki, k in enumerate(Ks):
        if partitions_unknown:
            partition_vec, Hnorm = find_partition(evecs, k, model, Dtau_sqrt_inv)
        else:
            partition_vec = partition_vecs[ki]
            Hnorm = create_normed_partition_matrix_from_vector(partition_vec, model)

        # calculate and store error
        error[ki] = calculate_proj_error(evecs, Hnorm, norm)
        # print("K, error ")
        # print(k, error[ki])

        # save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs


def find_partition(evecs, k, model, Dtau_sqrt_inv, method='GMMspherical', n_init=20):
    """ Perform clustering in spectral embedding space according to various criteria"""
    V = evecs[:, :k]

    if model == 'DCSBM':
        X = preprocessing.normalize(V, axis=1, norm='l2')
    elif model == 'SBM':
        # X = Dtau_sqrt_inv* V
        X = V
    else:
        error('something went wrong. Please specify valid mode')

    # select methof of clustering - QR, KM (k-means), or GMM
    if method == 'QR':
        partition_vec = clusterEVwithQR(X)
    elif method == 'KM':
        clust = KMeans(n_clusters=k, n_init=n_init)
        clust.fit(X)
        partition_vec = clust.labels_
        partition_vec = relabel_partition_vec(partition_vec)
    elif method == 'GMM':
        GMM = mixture.GaussianMixture(n_components=k, covariance_type='full', n_init=n_init)
        GMM.fit(X)
        partition_vec = GMM.predict(X)
        partition_vec = relabel_partition_vec(partition_vec)
    elif method == 'GMMspherical':
        GMM = mixture.GaussianMixture(n_components=k, covariance_type='spherical', n_init=n_init)
        GMM.fit(X)
        partition_vec = GMM.predict(X)
        partition_vec = relabel_partition_vec(partition_vec)
    elif method == 'GMMdiag':
        GMM = mixture.GaussianMixture(n_components=k, covariance_type='diag', n_init=n_init)
        GMM.fit(X)
        partition_vec = GMM.predict(X)
        partition_vec = relabel_partition_vec(partition_vec)
    else:
        raise ValueError('something went wrong. Please specify valid clustering method')

    H = create_normed_partition_matrix_from_vector(partition_vec, model)

    return partition_vec, H
