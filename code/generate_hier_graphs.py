#!/usr/bin/env python
import scipy
import numpy as np
from HierarchicalGraph import HierarchicalGraph

from cluster import Hierarchy
from cluster import Partition

def generateNetwork(hier_graph):
    """
    Network nodes at each leaf of the dendro are equivalent.  For each leaf work out the
    probability of connection with other blocks by working up to the root of the tree.
    """
    Omega = hier_graph.Omega
    pvec = hier_graph.get_partition_at_level(-1).pvec
    nc = [sum(pvec == i) for i in range(pvec.max() + 1)]
    A = sample_block_model(Omega,nc)
    return A

def sample_block_model(omega, nc, mode = 'undirected'):
    """
    Sample from a blockmodel, given the affinity matrix omega, and the number of nodes
    per group.

    Input:
        omega -- affinity matrix
        nc -- number of nodes per group

    Output:
        A -- sampled adjacency matrix
    """

    ni, nj = omega.shape

    # Question -- can we speed this up by collecting just the indices,
    # then creating a single sparse matrix at the end instead of concat
    if mode == 'undirected':
        if not np.all(omega.T == omega):
            raise Exception("You should provide a symmetric affinity matrix")
        # rowlist is a list of matrices to stack vertically
        rowlist = []
        for i in range(ni):
            # collist is a list of matrices to stack horizontally
            collist = []
            for j in range(nj):
                if i <= j:
                    pij = omega[i,j]
                else: 
                    pij = 0
                A1  = sample_from_block(nc[i],nc[j],pij)
                if i == j:
                    A1 = scipy.sparse.triu(A1,k=1)
                collist.append(A1)

            A1 = scipy.sparse.hstack(collist)
            rowlist.append(A1)

            A = scipy.sparse.vstack(rowlist)
        A = A + A.T

    elif mode != "undirected":
        raise NotImplementedError("sampling from asymmetric networks is not implemented yet")

    return A

def sample_from_block(m,n,pij):
    if pij == 0:
        block = scipy.sparse.csr_matrix((m,n))
    else:
        block = (scipy.sparse.random(m,n,density=pij) >0)*1
    return block


##################################
# SNR calculations for GHRG graphs
##################################

def calculateDegreesFromSNR(snr,ratio=0.5,num_cluster=2):
    """
    Given a particular signal to noise ratio (SNR), a ratio of in- vs out-link probabilities and a number of clusters,
    compute the degree parameters for a planted partition model.

    Output:  degree parameters a, b, such that the probability for an 'inside' connection is a/n, and for an outside connection b/n.
    """
    # SNR a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b)
    # fix SNR and b =r*a
    # SNR = a^2 *(1-r)^2 / (ka + k(k-1)*ra)
    # SNR = a * (1-r)^2 / (k + k(k-1)*r)
    # a = SNR * (k + k(k-1)*r) / (1-r)^2

    a = snr * (num_cluster + num_cluster*(num_cluster-1)*ratio) / float((1-ratio)**2)
    b = ratio*a;

    return a, b

def calculateDegreesFromAvDegAndSNR(SNR,av_degree,num_cluster=2):
    """
    Given a particular signal to noise ratio (SNR), the average degree and a number of clusters,
    compute the degree parameters for a planted partition model.

    Output:  degree parameters a, b, such that the probability for an 'inside' connection is a/n, and for an outside connection b/n.
    """
    amb = num_cluster * np.sqrt(av_degree*SNR)
    b = av_degree - amb/float(num_cluster)
    a = amb + b

    # SNR, a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b) = (a-b)^2 / [k^2 *av_degree]
    # av_degree = a/k + (k-1)*b/k = a-b /k + b

    return a, b


##############################
# SPECIFIC GRAPH CONSTRUCTIONS
##############################

def create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level):
    """
    Function to create a test GHRG for simulations.
    Parameters:
        n   : number of nodes
        snr : signal to noise ratio (1 represents theoretical detectability threshold)
        c_bar : average degree
        n_levels    : depth of GHRG
        groups_per_level     : number of groups at each level
    """

    omega=[]
    partitions = []
    n_this_level = n
    for level in range(n_levels):
        cin, cout = calculateDegreesFromAvDegAndSNR(snr,c_bar,groups_per_level)
        print("Hierarchy Level: ", level, '| KS Detectable: ', snr >=1, "| Link Probabilities in / out per block: ", cin/n_this_level,cout/n_this_level)

        # Omega is assigned on a block level, i.e. for each level we have one omega array
        # this assumes a perfect hierarchy with equal depth everywhere
        omega_level = np.ones((groups_per_level,groups_per_level))*cout/n_this_level + np.eye(groups_per_level)*(cin/n_this_level-cout/n_this_level)
        omega.append(omega_level)
        if np.any(omega[level]>=1):
            print("no probability > 1 not allowed")
            raise ValueError("Something wrong")

        num_current_groups = groups_per_level**(level+1)
        pvec = np.kron(np.arange(num_current_groups,dtype=int),np.ones(int(n/num_current_groups),dtype=int))
        partitions.append(pvec)

        n_this_level = n_this_level / groups_per_level
        if np.floor(n_this_level) != n_this_level:
            print("Rounding number of nodes")

        c_bar=(cin/n_this_level)*(n_this_level / groups_per_level)
    
    matrix = omega[0]
    for level in range(n_levels-1):
        Omega = matrix_fill_in_diag_block(omega[level+1],matrix)
        matrix = Omega

    # Q: should we adjust the Hierarchy constructors to make this less cumbersome
    Hier = Hierarchy(Partition(partitions.pop(0)))
    for pvec in partitions:
        Hier.append(Partition(pvec))
    
    graph = HierarchicalGraph(Hier, Omega)

    return graph

def createAsymGHRG(n,snr,c_bar,n_levels,groups_per_level):
    """
    Function to create an asymmetric test GHRG for simulations
    Parameters:
        n   : number of nodes
        snr : signal to noise ratio (1 represents theoretical detectability threshold)
        c_bar : average degree
        n_levels    : depth of GHRG
        groups_per_level     : number of groups at each level
    """

    omega=[]
    partitions = []
    n_this_level = n
    for level in range(n_levels):
        cin, cout = calculateDegreesFromAvDegAndSNR(snr,c_bar,groups_per_level)
        print("Hierarchy Level: ", level, '| KS Detectable: ', snr >=1, "| Link Probabilities in / out per block: ", cin/n_this_level,cout/n_this_level)

        # Omega is assigned on a block level, i.e. for each level we have one omega array
        # this assumes a perfect hierarchy with equal depth everywhere
        omega_level = np.ones((groups_per_level,groups_per_level))*cout/n_this_level + np.eye(groups_per_level)*(cin/n_this_level-cout/n_this_level)
        omega.append(omega_level)
        if np.any(omega[level]>=1):
            print("no probability > 1 not allowed")
            raise ValueError("Something wrong")


        n_this_level = n_this_level / float(groups_per_level)
        if np.floor(n_this_level) != n_this_level:
            print("Rounding number of nodes")
        c_bar=(cin/n_this_level)*(n_this_level / float(groups_per_level))

    start_partition = np.zeros(n,dtype=int)
    for level in range(n_levels):
        last_group_index = start_partition.max()
        nodes_in_last_group = np.nonzero(start_partition == last_group_index)[0]
        num_nodes_in_last_group = nodes_in_last_group.shape[0]
        start_partition[nodes_in_last_group] = np.kron(last_group_index+np.arange(groups_per_level,dtype=int),np.ones(int(num_nodes_in_last_group/groups_per_level),dtype=int))
        partitions.append(np.array(start_partition))

    matrix = omega[0]
    for level in range(n_levels-1):
        fill_in = [np.array([matrix[i,i]]) for i in range(matrix.shape[0]-1)]
        fill_in.append(omega[level+1])
        Omega = matrix_fill_in_diag_block(fill_in,matrix,same_diag=False)
        matrix = Omega

    # Q: should we adjust the Hierarchy constructors to make this less cumbersome
    Hier = Hierarchy(Partition(partitions.pop(0)))
    for pvec in partitions:
        Hier.append(Partition(pvec))
    
    graph = HierarchicalGraph(Hier, Omega)

    return graph

def matrix_fill_in_diag_block(diaA,B,same_diag=True):
    """
    Replace diagonal entries of a matrix by a set of blocks and expand off-diagonal entries
    accordingly.
    If same_diag=True replace each diagonal by the same block
    If same_diag=False input must be a list with one matrix for each diagonal entry
    """
    if same_diag:
        n, m = B.shape
        n2, m2 = diaA.shape
        if n != m or n2 != m2:
            raise Exception("Matrices should be square")
        D = np.kron(np.eye(n),diaA)
        B = np.kron(B,np.ones((n2,n2)))
        B[D.nonzero()] = 0
        A = B+D

    else:
        n, m = B.shape
        assert n == m
        assert type(diaA) == list
        assert len(diaA) == n

        D = diaA[0]
        partition_size= [D.shape[0],]
        for i in range(m-1):
             D = scipy.linalg.block_diag(D,diaA[i+1])
             partition_size.append(diaA[i+1].shape[0])
        
        pvec = np.concatenate([i*np.ones(p,dtype=int) for (i,p) in enumerate(partition_size)])
        partition = Partition(pvec)
        H = partition.H
        Bexpanded = H*B*H.T
        Bexpanded[D.nonzero()] = 0
        A = D+Bexpanded

    return A



def createTiagoHierNetwork():
    a = 0.95
    b = 0.75
    c = 0.3
    d = 0.1
    nodes_per_group =50
    Omega = np.array([[a, b, b, 0, 0, 0, d, 0, 0, 0, 0, 0],
                        [0, a, b, 0, 0, 0, 0, d, 0, 0, 0, 0],
                        [0, 0, a, c, 0, 0, 0, 0, d, 0, 0, 0],
                        [0, 0, 0, a, b, b, 0, 0, 0, d, 0, 0],
                        [0, 0, 0, 0, a, b, 0, 0, 0, 0, d, 0],
                        [0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, d],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    Omega = Omega + Omega.T -np.diagflat(np.diag(Omega))
    p1 = Partition(np.kron(np.array([0,0,0,0,0,0,1,1,1,1,1,1]),np.ones((nodes_per_group,),dtype=int)))
    p2 = Partition(np.kron(np.array([0,0,0,1,1,1,2,2,2,3,3,3]),np.ones((nodes_per_group,),dtype=int)))
    p3 = Partition(np.kron(np.arange(12,dtype=int),np.ones((nodes_per_group,),dtype=int)))
    Hier = Hierarchy(p1)
    Hier.add_level(p2)
    Hier.add_level(p3)
    return HierarchicalGraph(Hier,Omega)