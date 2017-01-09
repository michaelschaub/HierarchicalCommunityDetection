#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.sparse
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
# from matplotlib import pyplot as plt


def spectral_partition(A, mode='Lap', num_groups=2):
    """ Perform one round of spectral clustering for a given network matrix A
    Inputs: A -- input adjacency matrix
            mode -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            num_groups -- in how many groups do we want to split the graph?
            (default: 2; set to -1 to infer number of groups from spectrum)

            Output: partition_vec -- clustering of the nodes
    """

    if   mode == "Lap":
        partition = regularized_laplacian_spectral_clustering(A,num_groups=num_groups)

    elif mode == "Bethe":
        partition = cluster_with_BetheHessian(A,num_groups=num_groups)

    elif mode == "NonBack":
        pass

    return partition


##########################################
# REGULARIZED SPECTRAL CLUSTERING (ROHE)
##########################################

def regularized_laplacian_spectral_clustering(A, num_groups=2, tau=-1):
    """
    Performs regularized spectral clustering based on Qin-Rohe 2013 using a normalized and
    regularized adjacency matrix (called Laplacian by Rohe et al)
    """

    #TODO: adjust for sparse matrices

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau==-1:
        # set tau to average degree
        tau = A.sum()/float(A.shape[0])

    Dtau = np.diagflat(A.sum(axis=1) + tau*np.eye(A.shape[0]))

    Dtau_sqrt_inv = scipy.linalg.solve(np.sqrt(Dtau),np.eye(A.shape[0]))
    L = Dtau_sqrt_inv.dot(A).dot(Dtau_sqrt_inv)

    # make L sparse to be used within the 'eigs' routine
    L = scipy.sparse.coo_matrix(L)

    # compute eigenvalues and eigenvectors (sorted according to smallest magnitude first)
    ev, evecs = scipy.sparse.linalg.eigsh(L,num_groups,which='LM')

    X = preprocessing.normalize(evecs, norm='l2')

    clust = KMeans(n_clusters = num_groups)
    clust.fit(X)
    partition_vector = clust.labels_
    print partition_vec


    return partition_vector

######################################
# BETHE HESSIAN CLUSTERING
######################################

def build_BetheHessian(A, r):
    """
    Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
    B = (r^2-1)*I-r*A+D
    """
    if ~scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csc_matrix(A)

    d = A.sum(axis=1).getA().flatten().astype(float)
    B = scipy.sparse.eye(A.shape[0]).dot(r**2 -1) -r*A +  scipy.sparse.diags(d,0)
    return B


def build_weighted_BetheHessian(A,r):
    """
    Construct weigthed Bethe Hessian as discussed in Saade et al.
    """
    if ~scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csc_matrix(A)

    # we are only interested in A^.2 (elementwise)
    A2data = A.data **2

    new_data = A2data / (r*r -A2data)
    A2 = scipy.sparse.csr_matrix((new_data,A.nonzero()),shape=A.shape)

    # diagonal matrix
    d = 1 + A2.sum(axis=1)
    d = d.getA().flatten()
    DD = scipy.sparse.diags(d,0)

    # second matrix
    rA_data = r*A.data / (r*r - A2data)
    rA = scipy.sparse.csr_matrix((rA_data,A.nonzero()))

    # full Bethe Hessian
    BHw = DD - rA
    return BHw


def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa'):
    """
    Perform one round of spectral clustering using the Bethe Hessian
    """

    if regularizer=='BHa':
        # set r to square root of average degree
        r = A.sum()/float(A.shape[0])
        r = np.sqrt(r)

    elif regularizer=='BHm':
        d = A.sum(axis=1).getA().flatten().astype(float)
        r = np.sum(d*d)/np.sum(d) - 1
        r = np.sqrt(r)

    # construct both the positive and the negative variant of the BH
    if not all(A.sum(axis=1)):
        print "GRAPH CONTAINS NODES WITH DEGREE ZERO"

    if all(A.sum(axis=1) == 0):
        print "empty Graph -- return all in one partition"
        partition_vector = np.zeros(A.shape[0],dtype='int')
        return partition_vector

    BH_pos = build_BetheHessian(A,r)
    BH_neg = build_BetheHessian(A,-r)
    # print "BHPOS"
    # print BH_pos.shape


    if num_groups ==-1:
        relevant_ev, _ = find_negative_eigenvectors(BH_pos)
        X = relevant_ev

        relevant_ev, _ = find_negative_eigenvectors(BH_neg)
        X = np.hstack([X, relevant_ev])
        #TODO: check if we want to sort here as well?!
        num_groups = X.shape[1]

        if num_groups == 0:
            print "no indication for grouping -- return all in one partition"
            partition_vector = np.zeros(A.shape[0],dtype='int')
            return partition_vector

    else:
        # find eigenvectors corresponding to the algebraically smallest (most neg.) eigenvalues
        ev_pos, evecs_pos = scipy.sparse.linalg.eigsh(BH_pos,num_groups,which='SA')
        ev_neg, evecs_neg = scipy.sparse.linalg.eigsh(BH_neg,num_groups,which='SA')
        ev_all = np.hstack([ev_pos, ev_neg])
        index = np.argsort(ev_all)
        X = np.hstack([evecs_pos,evecs_neg])
        X = X[:,index]


    clust = KMeans(n_clusters = num_groups)
    clust.fit(X)
    partition_vector = clust.labels_

    return partition_vector

#######################################################
# HELPER FUNCTIONS
#######################################################
def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    k=len(np.unique(partition_vec))

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix

def build_projector_matrix(pvector):
    """
    Build a projection matrix onto the space spanned by the partition indicator matrix as
    described by the input vector
    """
    Htemp = create_partition_matrix_from_vector(pvector)
    D = Htemp.T.dot(Htemp)
    P = Htemp.dot(scipy.sparse.linalg.spsolve(D,Htemp.T))
    return P


def find_negative_eigenvectors(M):
    """
    Given a matrix M, find all the eigenvectors associated to negative eigenvalues
    and return the tuple (evecs, evalus)
    """
    Kmax = M.shape[0]-1
    K = min(10,Kmax)
    ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
    relevant_ev = np.nonzero(ev <0)[0]
    while (relevant_ev.size  == K):
        K = min(2*K, Kmax)
        ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
        relevant_ev = np.nonzero(ev<0)[0]

    return evecs[:,relevant_ev], ev[relevant_ev]

############################
# OUTDATED??
############################


# still in use?
# def hier_cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa'):
    # """
    # Perform one round of spectral clustering using the Bethe Hessian
    # """

    # if regularizer=='BHa':
        # # set r to square root of average degree
        # r = A.sum()/float(A.shape[0])
        # r = np.sqrt(r)

    # elif regularizer=='BHm':
        # d = A.sum(axis=1).getA().flatten().astype(float)
        # r = np.sum(d*d)/np.sum(d) - 1
        # r = np.sqrt(r)

    # # construct both the positive and the negative variant of the BH
    # if not all(A.sum(axis=1)):
        # print "GRAPH CONTAINS NODES WITH DEGREE ZERO"

    # if all(A.sum(axis=1) == 0):
        # print "empty Graph -- return all in one partition"
        # partition_vector = np.zeros(A.shape[0],dtype='int')
        # return partition_vector

    # BH_pos = build_BetheHessian(A,r)
    # BH_neg = build_BetheHessian(A,-r)

    # if num_groups ==-1:
        # relevant_ev, evalues = find_negative_eigenvectors(BH_pos)
        # relevant_ev2, evalues2 = find_negative_eigenvectors(BH_neg)
        # X = np.hstack([relevant_ev, relevant_ev2])
        # all_ev = np.hstack([evalues, evalues2])

        # num_groups = X.shape[1]

        # ordering = np.argsort(all_ev)
        # evalues_sorted = all_ev[ordering]
        # relevant_ev_sorted = X[:,ordering]

        # if num_groups == 0:
            # print "no indication for grouping -- return all in one partition"
            # partition_vector = np.zeros(A.shape[0],dtype='int')
            # return partition_vector

    # else:
        # # find eigenvectors corresponding to the algebraically smallest (most neg.) eigenvalues
        # ev_pos, evecs_pos = scipy.sparse.linalg.eigsh(BH_pos,num_groups,which='SA')
        # ev_neg, evecs_neg = scipy.sparse.linalg.eigsh(BH_neg,num_groups,which='SA')
        # ev_all = np.hstack([ev_pos, ev_neg])
        # index = np.argsort(ev_all)
        # X = np.hstack([evecs_pos,evecs_neg])
        # X = X[:,index[:num_groups]]


    # clust = KMeans(n_clusters = num_groups)
    # clust.fit(X)
    # partition_vector = clust.labels_

    # for ii in range(1,num_groups):
        # temp_cluster= KMeans(n_clusters = ii+1)
        # temp_cluster.fit(X[:,:ii])
        # P = build_projector_matrix(temp_cluster.labels_)
        # delta = 0.9
        # test = np.linalg.norm(P.dot(X),axis=0)
        # print test > delta
        # # from IPython import embed
        # # embed()

    # return partition_vector


# def hier_spectral_partition(A, mode='Lap', num_groups=2):
    # """ Perform one round of spectral clustering for a given network matrix A
    # Inputs: A -- input adjacency matrix
            # mode -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            # num_groups -- in how many groups do we want to split the graph?
            # (default: 2; set to -1 to infer number of groups from spectrum)

            # Output: partition_vec -- clustering of the nodes
    # """

    # if   mode == "Lap":
        # print "NOT Implemented yet"
        # pass

    # elif mode == "Bethe":
        # partition = hier_cluster_with_BetheHessian(A,num_groups=num_groups)

    # elif mode == "NonBack":
        # pass

    # return partition
