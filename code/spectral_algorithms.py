#!/usr/bin/env python

import numpy as np
import scipy.sparse
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import defaultdict
from GHRGmodel import GHRG


def split_network_by_recursive_spectral_partition(A, mode='Lap', num_groups=2, max_depth=3):
    """ Recursively split graph into pieces by employing a spectral clustering strategy.

    Inputs: A          -- input adjacency matrix
            mode       -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            num_groups -- in how many groups do we want to split the graph at each step
                          (default: 2; set to -1 to infer number of groups from spectrum)
            max_depth  -- how many times do we want to recursively split the graph (default:3)
                          Set to -1 for partitioning graph completely


            Output: networkx dendrogram
    """

    nr_nodes = A.shape[0]
    current_partition = spectral_partition(A, mode=mode, num_groups=num_groups)

    # initialise networkx output dendrogram, and store some things as properties of the graph
    Dendro = GHRG()
    Dendro.network_nodes = np.arange(nr_nodes)
    Dendro.root_node = 0

    # create root node and assign properties
    Emat, Nmat = compute_number_links_between_groups(A,current_partition)
    Dendro.add_node(Dendro.root_node, Er=Emat, Nr=Nmat)
    # names of nodes corresponding to node in Dendrogram
    Dendro.node[Dendro.root_node]['nnodes'] = Dendro.network_nodes
    # number of nodes corresponding to this node
    Dendro.node[Dendro.root_node]['n'] = nr_nodes

    nr_groups = current_partition.max()+1
    nodes_next_level = Dendro.add_children(Dendro.root_node, nr_groups)
    Dendro.node[Dendro.root_node]['children'] = nodes_next_level
    for i, n in enumerate(nodes_next_level):
        subpart = current_partition == i
        Dendro.node[n]['nnodes'] = subpart.nonzero()[0]
        Dendro.node[n]['n'] = len(subpart.nonzero()[0])

    hier_depth = 0
    print "\nNow running recursion"

    # as long as we have not reached the max_depth yet,
    # and there is more than one group in the partition
    while (hier_depth < max_depth or max_depth == -1) and len(nodes_next_level):

        #~ print "\nLEVEL"
        #~ print nodes_next_level
        next_level_temp = []

        for node in nodes_next_level:
            # print "\nNODE"
            # print node

            # create subgraphs
            subpart = Dendro.node[node]['nnodes']
            Asub = A[subpart,:]
            Asub = Asub[:,subpart]
            # print Asub
            # print "SUBPART"
            # print subpart


            # cluster subgraph recursively
            partition = spectral_partition(Asub, mode=mode, num_groups=num_groups)
            # print "PARTITION"
            # print partition

            Emat, Nmat = compute_number_links_between_groups(Asub,partition)
            Dendro.node[node]['Er'] = Emat
            Dendro.node[node]['Nr'] = Nmat
            # print "EMAT"
            # print Emat
            # print Dendro.node[0]['Er']
            nr_groups = np.unique(partition).size

            # print "NRG"
            # print nr_groups
            if nr_groups > 1:
                children  = Dendro.add_children(node,nr_groups)
                Dendro.node[node]['children'] = children
                next_level_temp.extend(children)
                parent_nnodes = Dendro.node[node]['nnodes']
                for i, n in enumerate(children):
                    subpart = partition == i
                    Dendro.node[n]['nnodes'] = parent_nnodes[subpart.nonzero()[0]]
                    Dendro.node[n]['n'] = len(subpart.nonzero()[0])
            else:
                Dendro.node[node]['children'] = []


        nodes_next_level = next_level_temp
        hier_depth +=1

    return Dendro

def compute_number_links_between_groups(A,partition_vec):
    """
    Compute the number of possible and actual links between the groups indicated in the
    partition vector
    """

    pmatrix = create_partition_matrix_from_vector(partition_vec)
    # number of columns is number of groups
    nr_groups = pmatrix.shape[1]

    links_between_groups = pmatrix.T.dot(A).dot(pmatrix).toarray()

    nodes_per_group = pmatrix.sum(0).getA()
    possible_links_between_groups = np.outer(nodes_per_group,nodes_per_group)

    return links_between_groups, possible_links_between_groups

def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    k=len(np.unique(partition_vec))

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix

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


def regularized_laplacian_spectral_clustering(A, num_groups=2, tau=-1):
    """
    Performs regularized spectral clustering based on Qin-Rohe 2013 using a normalized and
    regularized adjacency matrix (called Laplacian by Rohe et al)
    """

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau==-1:
        # set tau to average degree
        tau = np.sum(A)/float(A.shape[0])

    Dtau = np.diagflat(np.sum(A,axis=1)) + tau*np.eye(A.shape[0])

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


    return partition_vector

def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa'):
    """
    Perform one round of spectral clustering using the Bethe Hessian
    """
    def find_negative_eigenvectors(M):
        Kmax = M.shape[0]-1
        K = min(10,Kmax)
        ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
        relevant_ev = np.nonzero(ev <0)[0]
        while (relevant_ev.size  == K):
            K = min(2*K, Kmax)
            ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
            relevant_ev = np.nonzero(ev<0)[0]

        return evecs[:,relevant_ev]

    def find_enough_eigenvectors(M):
        Kmax = M.shape[0]-1
        K = min(30,Kmax)
        ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
        neg_ev = np.nonzero(ev <0)[0]
        while (neg_ev.size  == K):
            K = min(2*K, Kmax)
            ev, evecs = scipy.sparse.linalg.eigsh(M,K,which='SA')
            neg_ev = np.nonzero(ev<0)[0]

        return ev, evecs

    # check if tau regularisation parameter is specified otherwise go for mean degree...
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

    if r <= 0:
        print "Something is going wrong here, regularizer should be greater than 0"

    BH_pos = build_BetheHessian(A,r)
    BH_neg = build_BetheHessian(A,-r)
    # print "BHPOS"
    # print BH_pos.shape


    if num_groups ==-1:
        relevant_ev = find_negative_eigenvectors(BH_pos)
        X = relevant_ev

        relevant_ev = find_negative_eigenvectors(BH_neg)
        X = np.hstack([X, relevant_ev])
        num_groups = X.shape[1]

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


def build_BetheHessian(A, r):
    """
    Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
    B = (r^2-1)*I-r*A+D
    """
    d = A.sum(axis=1).getA().flatten().astype(float)
    B = scipy.sparse.eye(A.shape[0]).dot(r**2 -1) -r*A +  scipy.sparse.diags(d,0)
    return B


def build_weighted_BetheHessian(A,r):
    """
    Construct weigthed Bethe Hessian as discussed in Saade et al. TODO
    """
    pass


def create_block_model_graph(affinity_matrix, nr_nodes=None, partition_vec=None):
    # affinity matrix, partition_vec and degree_seq should all be numpy arrays
    # UNCORRECTED SBM -- affinity matrix should specify the link probability between each block.

    if nr_nodes is None and partition_vec is None:
        print "Error -- you should either give the number of nodes or a valid partition"
        return
    elif nr_nodes is not None and partition_vec is not None:
        print "Error -- you should either give the number of nodes *or* a valid partition"
        return
    if nr_nodes is None:
        nr_nodes = partition_vec.size
        if ( not (unique(partition_vec)==arange(affinity_matrix.shape[0])).all() ) or \
             len(unique(partition_vec)) != affinity_matrix.shape[0]:
            print "partition vector, number of nodes and affinity matrix should match"
            return
    elif partition_vec is None:
        nr_groups = affinity_matrix.shape[0]
        group_size = nr_nodes / nr_groups
        rem = nr_nodes % nr_groups
        partition_vec = np.hstack([np.zeros((1,rem)),
                   np.kron(np.arange(nr_groups),np.ones((1,group_size)))]).flatten()
        #print partition_vec



    partition_matrix = create_partition_matrix_from_vector(partition_vec)
    P = partition_matrix.dot(affinity_matrix).dot(partition_matrix.T)

    A = P > np.random.rand(nr_nodes,nr_nodes)

    return 1*A

def create_example_graph():
    # create hier. graph with 3 layers partition into (8,4,2) groups.
    p_step_innermost =  0.3
    p_step_second = 0.3
    p_step_third = 0.3

    smallest_group_size = 50
    nodes_total = smallest_group_size*2*2*2

    p_rand = 2./nodes_total

    # assemble Prob matrix
    P_temp = np.kron(np.eye(8), p_step_innermost*np.ones((smallest_group_size,smallest_group_size)))
    P_temp2 = np.kron(np.eye(4), p_step_second*np.ones((2*smallest_group_size,2*smallest_group_size)))
    P_temp3 = np.kron(np.eye(2), p_step_third*np.ones((2*2*smallest_group_size,2*2*smallest_group_size)))
    P_outer  = np.kron(np.ones((2,2))-np.eye(2), p_rand*np.ones((4*smallest_group_size,4*smallest_group_size)))

    P_tot = P_temp + P_temp2 + P_temp3 + P_outer
    A = P_tot > np.random.rand(nodes_total,nodes_total)

    A = 1*scipy.sparse.tril(A) + scipy.sparse.tril(A).T - np.diag(np.diag(A))
    return A

if __name__ == "__main__":

    A = create_example_graph()
    Dendro = split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)
    print Dendro.nodes()
