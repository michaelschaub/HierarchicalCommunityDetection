#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as linalg
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt


def spectral_partition(A, mode='Lap', num_groups=2):
    """ Perform one round of spectral clustering for a given network matrix A
    Inputs: A -- input adjacency matrix
            mode -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            num_groups -- in how many groups do we want to split the graph?
            (default: 2; set to -1 to infer number of groups from spectrum)

            Output: partition_vec -- clustering of the nodes
    """

    if   mode == "Lap":
        partition, _ = regularized_laplacian_spectral_clustering(A,num_groups=num_groups)

    elif mode == "Bethe":
        partition = cluster_with_BetheHessian(A,num_groups=num_groups)

    elif mode == "NonBack":
        pass

    else:
        raise ValueError("mode '%s' not recognised - available modes are 'Lap', Bethe', or 'NonBack'" % mode)

    partition = relabel_partition_vec(partition)
    return partition


##########################################
# REGULARIZED SPECTRAL CLUSTERING (ROHE)
##########################################

def regularized_laplacian_spectral_clustering(A, num_groups=2, tau=-1):
    """
    Performs regularized spectral clustering based on Qin-Rohe 2013 using a normalized and
    regularized adjacency matrix (called Laplacian by Rohe et al)
    """

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau==-1:
        # set tau to average degree
        tau = A.sum()/A.shape[0]


    Dtau_sqrt_inv = scipy.sparse.diags(np.power(np.array(A.sum(1)).flatten() + tau,-.5),0)
    L = Dtau_sqrt_inv.dot(A).dot(Dtau_sqrt_inv)


    # compute eigenvalues and eigenvectors (sorted according to smallest magnitude first)
    ev, evecs = scipy.sparse.linalg.eigsh(L,num_groups,which='LM')

    X = preprocessing.normalize(evecs, axis=1, norm='l2')

    clust = KMeans(n_clusters = num_groups)
    clust.fit(X)
    partition_vector = clust.labels_


    return partition_vector, evecs

######################################
# BETHE HESSIAN CLUSTERING
######################################

def build_BetheHessian(A, r):
    """
    Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
    B = (r^2-1)*I-r*A+D
    """
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)

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
        r = A.sum()/A.shape[0]
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

##########################################
# NON-BACKTRACKING matrix
##########################################

def build_non_backtracking_matrix(A,mode='unweighted'):
    """Build non-backtracking matrix as defined in Krzakala et al 2013:
    Starting from a similarity matrix (adjacency) matrix s(u,v), we have
         B(u>v;w>x) = s(u,v) if v = w and u != x, and 0 otherwise
            (weighted_end setting, column weighting)
         B(u>v;w>x) = s(w,x) if v = w and u != x, and 0 otherwise
            (weighted_start setting, row weighting)
    """
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)

    edgelist = A.nonzero()
    weights = A.data
    number_edges = weights.size

    start_node = edgelist[0]
    end_node = edgelist[1]

    NodeToEdgeIncidenceMatrixStart = scipy.sparse.csr_matrix((np.ones_like(start_node),(start_node,np.arange(number_edges))))
    NodeToEdgeIncidenceMatrixEnd =  scipy.sparse.csr_matrix((np.ones_like(end_node),(end_node,np.arange(number_edges))))

    # Line Graph connecting all edge points with start points
    BT = NodeToEdgeIncidenceMatrixEnd.T*NodeToEdgeIncidenceMatrixStart

    # Backtracking links are the only ones that are symmetric
    BT = BT - BT.multiply(BT.T)

    if mode == 'weighted_start':
        BT = scipy.sparse.diags(weights,0)*BT
    elif mode == 'weighted_end':
        BT = BT*scipy.sparse.diags(weights,0)
    elif mode != 'unweighted':
        print "no valid mode specified"
        return -1

    return BT

#################################
# GAUSSIAN MATRIX CLUSTERING
#################################

def find_relevant_eigenvectors_Gaussian(Omega):
    """
    Given a matrix of normalized edge counts between the groups, and assuming that these
    edge-counts are derived from an undirected network, we interpret the resulting reducing array of edge_counts (made mean-free and normalized) as an array of Gaussians, and use RMT to find the expected maximal eigenvalue.
    """
    #TODO
    pass


##################################################
# SPECTRAL MODEL SELECTION VIA INVARIANT SUBSPACE
##################################################

def identify_hierarchy_in_affinity_matrix_DCSBM(Omega,mode='DCSBM',reg=True):

    max_k = Omega.shape[0]
    best_k = max_k
    error = 999

    if reg:
        # set tau to average degree
        tau = Omega.sum()/Omega.shape[0]
    else:
        tau = 0

    # construct Laplacian
    Dtau_sqrt_inv = scipy.sparse.diags(np.power(np.array(Omega.sum(1)).flatten() + tau,-.5),0)
    # print Omega
    L = Dtau_sqrt_inv.dot(Omega)
    L = Dtau_sqrt_inv.dot(L.T).T
    L = (L+L.T)/2

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    # print L

    for k in xrange(max_k-1,0,-1):
        if mode == 'DCSBM':
            V = evecs[:,:k]
            # print V
            X = preprocessing.normalize(V, axis=1, norm='l2')
            clust = KMeans(n_clusters = k)
            clust.fit(X)
            partition_vec = clust.labels_
            partition_vec = relabel_partition_vec(partition_vec)
            # print partition_vec

            H = create_partition_matrix_from_vector(partition_vec)
            # Dsqrt = scipy.sparse.diags(scipy.sqrt(Omega.sum(axis=1)+tau).flatten())
            H = Dtau_sqrt_inv.dot(H)
            H = preprocessing.normalize(H,axis=0,norm='l2')

        elif mode == 'SBM':
            pass
            #TODO: corresponding SBM version of reg. clustering
        else:
            error('something went wrong. please specify valid mode')

        proj1 = project_orthogonal_to(H,V)
        proj2 = project_orthogonal_to(V,H)
        norm1 = scipy.linalg.norm(proj1)
        norm2 = scipy.linalg.norm(proj2)
        error = 0.5*(norm1+norm2)
        # print k, error
        # Note that this should always be fulfilled at k=1
        if error < 0.01*max_k:
            return k, partition_vec, H, error

def project_orthogonal_to(subspace_basis,vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space orthogonal to which we want to project
    vectors_to_project: project these vectors into the orthogonal complement of the
    specified subspace
    """
    if scipy.sparse.issparse(vectors_to_project):
        vectors_to_project = vectors_to_project.toarray()
    projected = subspace_basis.T.dot(vectors_to_project)
    normalization = subspace_basis.T.dot(subspace_basis)
    if scipy.sparse.issparse(normalization):
        normalization  = normalization.toarray()
    normalization_inv = scipy.linalg.solve(normalization,scipy.eye(normalization.shape[1]))
    projected = subspace_basis.dot(normalization_inv.dot(projected))

    orthogonal_proj = vectors_to_project - projected
    return orthogonal_proj



#######################################################
# HELPER FUNCTIONS
#######################################################
def relabel_partition_vec(pvec):
    k = pvec.max()+1
    remap = -np.ones(k,dtype='int')
    new_id = 0
    for element in pvec:
        if remap[element] == -1:
            remap[element] = new_id
            new_id += 1
    pvec = remap[pvec]
    return pvec


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

def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    k=len(np.unique(partition_vec))

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix

#TODO: implement method for comparison
def find_relevant_eigenvectors_Le_Levina(M, multiplier=5):
    pass
