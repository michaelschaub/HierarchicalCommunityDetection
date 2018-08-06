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

"""
Helper functions collects a bunch of useful small utility functions that are used in various places elsewhere
"""


#######################################################
# DEALING WITH PARTITION VECTORS AND PARTITION MATRICES
#######################################################

def relabel_partition_vec(pvec):
    """
    Given a partition vectors pvec, relabel the groups such that the new partition vector has contiguous group labels (starting with 0)
    """
    k = pvec.max()+1 # group labels start with 0
    if k==1:
        return pvec
    else:
        remap = -np.ones(k,dtype='int')
        new_id = 0
        for element in pvec:
            if remap[element] == -1:
                remap[element] = new_id
                new_id += 1
        pvec = remap[pvec]
        return pvec

def expand_partitions_to_full_graph(pvecs):
    """
    Map list of aggregated partition vectors to list of full-sized partition vectors
    """

    # partiitions are stored relative to the size of the aggregated graph, so we have to
    # expand them again into the size of the full graph

    # the finest partition is already at the required size
    pvec_new = []
    pvec_new.append(pvecs[0])


    # loop over all other partition
    for i in xrange(len(pvecs)-1):
        # get the partition from the previous level
        p_full_prev_level = pvec_new[i]

        # get aggregated partition from this level
        p_agg_this_level = pvecs[i+1]

        # group indices of previous level correspond to nodes in the aggregated graph;
        # get the group ids of those nodes, and expand by reading out one index per
        # previous node
        partition = p_agg_this_level[p_full_prev_level]
        pvec_new.append(partition)

    return pvec_new

def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    k=len(np.unique(partition_vec))

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix

def create_normed_partition_matrix_from_vector(partition_vec,mode):
    """
    Create a normalized partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.

    The returned partition indicator matrix will have column norm 1, i.e., H^T*H = Id
    """

    H = create_partition_matrix_from_vector(partition_vec)

    if mode == 'DCSBM':
        #TODO -- check this part
        print "TODO -- check this normalization"
        Dsqrt = scipy.sparse.diags(scipy.sqrt(Omega.sum(axis=1)+tau).flatten())
        H = Dtau_sqrt.dot(H)

    # normalize column norm to 1 of the partition indicator matrices
    return preprocessing.normalize(H,axis=0,norm='l2')



##########################################
# EIGENVALUE / EIGENVECTOR SEARCH ROUTINES
##########################################

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

def find_relevant_eigenvectors_Le_Levina(A, t=5):
    """ Find the relevant eigenvectors (of the Bethe Hessian) using the criteria proposed
        by Le and Levina (2015)
        """
    # start by computing first Kest eigenvalues/vectors
    Kest_pos = 10
    if Kest_pos > A.shape[0]:
        Kest_pos = A.shape[0]
    ev_BH_pos, evecs_BH_pos = scipy.sparse.linalg.eigsh(A,Kest_pos,which='SA')
    relevant_ev = np.nonzero(ev_BH_pos <=0)[0]
    while (relevant_ev.size  == Kest_pos):
        Kest_pos *=2
        if Kest_pos > A.shape[0]:
            Kest_pos = A.shape[0]
        # print Kest_pos.shape
        # print BH_pos.shape
        ev_BH_pos, evecs_BH_pos = scipy.sparse.linalg.eigsh(A,Kest_pos,which='SA')
        relevant_ev = np.nonzero(ev_BH_pos <=0)[0]

    ev_BH_pos.sort()
    tev = t*ev_BH_pos
    kmax = 0
    for k in range(ev_BH_pos.size-1):
        if tev[k] <= ev_BH_pos[k+1]:
            kmax = k+1
        else:
            break

    X = evecs_BH_pos[:,range(kmax)]

    return ev_BH_pos[:kmax], X

###################################################
# COMPUTING WITH 'AGGREGATED'/ PARTITIONED MATRICES
###################################################

def compute_number_links_between_groups(A,partition_vec,directed=True):
    """
    Compute the number of possible and actual links between the groups indicated in the
    partition vector.
    """
    #TODO: option to declare whether self-loops should be accounted for!?

    pmatrix = create_partition_matrix_from_vector(partition_vec)

    if not scipy.sparse.issparse(A):
        A = scipy.mat(A)

    # all inputs are matrices here!
    # calculation works accordingly and transforms to array only afterwards
    # each block counts the number of half links / directed links
    links_between_groups = pmatrix.T * A * pmatrix
    links_between_groups = links_between_groups.A

    if not directed:
        links_between_groups = links_between_groups - np.diag(np.diag(links_between_groups))/2.0
        links_between_groups = np.triu(links_between_groups)

    # convert to array type first, before performing outer product
    nodes_per_group = pmatrix.sum(0).A
    possible_links_between_groups = np.outer(nodes_per_group, nodes_per_group)

    if not directed:
        possible_links_between_groups = possible_links_between_groups - np.diag(nodes_per_group.flatten())
        possible_links_between_groups = possible_links_between_groups - np.diag(np.diag(possible_links_between_groups))/2.0
        possible_links_between_groups = np.triu(possible_links_between_groups)


    return links_between_groups, possible_links_between_groups


def compute_likelihood_SBM(pvec,A,omega=None):
    """Compute log-likelihood of SBM for a given partition vector"""

    def xlogy(x,y):
        """Compute x log(y) elementwise, with the convention that 0log0 = 0"""
        xlogy = x*np.log(y)
        xlogy[np.isinf(xlogy)] = 0
        xlogy[np.isnan(xlogy)] = 0
        return xlogy

    H = create_partition_matrix_from_vector(pvec)
    # self-loops and directedness is not allowed here
    Emat, Nmat = compute_number_links_between_groups(A,pvec,directed=False)
    if omega is None:
        omega = Emat / Nmat

    logPmat = xlogy(Emat,omega) + xlogy(Nmat-Emat,1 - omega)
    likelihood = logPmat.sum()
    return likelihood


def add_noise_to_small_matrix(M,snr=0.001,noise_type="gaussian"):
    """Add some small random noise to a (dense) small matrix as a perturbation"""

    #noise level is taken relative to the Froebenius norm
    normM = scipy.linalg.norm(M)

    if noise_type == "uniform":
        #TODO -- should we have uniform noise?
        pass
    elif noise_type == "gaussian":
        n,m = M.shape
        noise = scipy.triu(scipy.random.randn(n,m))
        noise = noise + noise.T -scipy.diag(scipy.diag(noise))
        normNoise = scipy.linalg.norm(noise)
        Mp = M + snr*normM/normNoise*noise

    return Mp

##################################################
# VARIOUS OTHER THINGS
##################################################


def test_sparse_and_transform(A):
    """ Check if matrix is sparse and if not, return it as sparse matrix"""
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)
    return A


def find_local_minima(vec):
    """find local minimum in a vector"""
    #TODO: check if this function can be replaced by argrelmin of scipy...

    #difference of errors err_k - err_{k+1}
    vec_diff = np.copy(vec)
    vec_diff[1:] -= vec_diff[:-1]

    #find sign of vector
    sign = np.sign(vec_diff)
    #shift 0's (no difference) to positive
    sign[sign==0] = 1

    sign_diff = np.diff(sign)
    goes_neg = (sign_diff==-2).nonzero()[0]+1
    goes_pos = (sign_diff==2).nonzero()[0]+1

    print "VEC", vec
    print vec_diff
    print sign_diff
    print goes_neg
    print goes_pos

    #sometimes there are no sign changes
    try:
        if goes_neg[0]<goes_pos[0]:
            segments = zip(goes_neg, goes_pos)
            if len(goes_neg)>len(goes_pos):
                segments.append((goes_neg[-1],len(vec)))
        else:
            segments = zip(np.append(0,goes_neg), goes_pos)
    #catches no sign or single sign change
    except IndexError:
        print "IndexError dues to <=1 sign change"
        #check if the minimum corresponds to a local minimum
        #i.e. does min of vec correspond to -ve in vec_diff
        if vec_diff[np.argmin(vec)]<0:
            return np.array([np.argmin(vec)])
        #otherwise return empty set
        else:
            return np.array([],dtype=int)

    minima=[]
    print "SEG", segments

    for seg in segments:
        minima.append(seg[0]+np.argmin(vec[seg[0]:seg[1]]))

    return np.array(minima)
