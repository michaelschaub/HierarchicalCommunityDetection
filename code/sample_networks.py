from __future__ import division
from GHRGmodel import GHRG
import numpy as np
import scipy
from scipy import linalg
import random

################
## DEPRECATED: functions moved to GHRGbuild module (to be deleted later)
def calculateDegreesFromSNR(snr,ratio=0.5,num_cluster=2):
    print "DEPRECATED: function moved to GHRGbuild module (to be deleted later)"
    # SNR a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b)
    # fix SNR and b =r*a
    # SNR = a^2 *(1-r)^2 / (ka + k(k-1)*ra)
    # SNR = a * (1-r)^2 / (k + k(k-1)*r)
    # a = SNR * (k + k(k-1)*r) / (1-r)^2
    a = snr * (num_cluster + num_cluster*(num_cluster-1)*ratio) / float((1-ratio)**2);
    b = ratio*a;

    return a, b

def calculateDegreesFromAvDegAndSNR(SNR,av_degree,num_cluster=2):
    print "DEPRECATED: function moved to GHRGbuild module (to be deleted later)"
    # SNR, a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b) = (a-b)^2 / [k^2 *av_degree]
    # av_degree = a/k + (k-1)*b/k = a-b /k + b
    amb = num_cluster * np.sqrt(av_degree*SNR)
    b = av_degree - amb/float(num_cluster)
    a = amb + b

    return a, b
################


def checkDetectabliityGeneralSBM(omega,nc):
    """
    Given a SBM with affinity matrix omega and group size distribuution nc, compute
    whether there is something detectable here.
    """

    # form matrix M
    M = omega.dot(np.diag(nc,0))
    u = scipy.linalg.eig(M)
    idx = u.argsort()[::-1]
    eigenvalues = u[idx]

    snr = eigenvalues[1]**2 / eigenvalues[0]

    return snr



def expand_partitions_to_full_graph(pvecs):
    pvec_new = []
    pvec_new.append(pvecs[0])

    for i in xrange(len(pvecs)-1):
        pold = pvecs[i]
        pnew = pvecs[i+1]
        partition = pnew[pold]
        pvec_new.append(partition)

    return pvec_new


def sample_hier_block_model(groups_per_level = np.array([2,4]), nnodes = 1000, av_deg=10, snr=8, group_sizes ='same'):
    """
    Function to sample from a hierarchical blockmodel with equal sized groups according
    to a given specification
    Inputs:
        groups_per_level -- number of splits in each level of the hierarchy

        nnodes -- number of nodes in the network total

        av_deg -- average degree in the network

        snr -- signal to noise ratio within each level

        group_sizes -- use equal sized groups or not; if not the snr is not meaningful any more..
    """


    # how many levels do we have and how many groups in total (lowest level)
    nr_levels = len(groups_per_level)
    if nr_levels < 2:
        raise Exception("no hierarchical structure, use different sampling function")
    nr_groups_til_level = np.cumprod(groups_per_level)
    nr_groups_total = nr_groups_til_level[-1]

    if group_sizes == 'same':
        nc = nnodes / nr_groups_total*np.ones(nr_groups_total)
    elif group_sizes == 'mixed':
        nc = nnodes / nr_groups_total*np.ones(nr_groups_total)
        min_size = nc[0] /4
        nc = nc + np.random.randint(-min_size,min_size,nc.shape)

    num_cluster = 1
    Omega = np.array([[1]])
    for level_ in xrange(nr_levels):

        # split nodes into groups and create vector of group sizes
        num_cluster*=groups_per_level[level_]
        nodes_per_group = nnodes / num_cluster*np.ones(num_cluster)

        # number of clusters at this level, and associated degree parameters
        num_cluster_this_level = groups_per_level[level_]
        a, b = calculateDegreesFromAvDegAndSNR(snr,av_deg,num_cluster_this_level)
        # at the next level up the average degree is equal to the degree on
        # the diagonal block
        av_deg = a / num_cluster_this_level


        # affinity matrix for this level
        omega = 1/float(nnodes)*((a-b)*np.eye(num_cluster_this_level) + b*np.ones((num_cluster_this_level,num_cluster_this_level)))
        # readjust normzlization of number of nodes -- should correspond to
        nnodes = nnodes / num_cluster_this_level
        Omega = matrix_fill_in_diag_block(omega,Omega)


    A = sample_block_model(Omega,nc, mode='undirected')
    pvecs = []
    nnodes = nc.sum()
    for i in range(groups_per_level.size):
        partition  = np.kron(np.arange(nr_groups_til_level[i]),np.ones(nnodes/nr_groups_til_level[i]))
        pvecs.append(partition)

    return A, pvecs

def matrix_fill_in_diag_block(diaA,B):
    """
    Replace diagonal entries of a matrix by a block and expand off-diagonal entries
    accordingly
    """
    n, m = B.shape
    n2, m2 = diaA.shape
    if n != m or n2 != m2:
        raise Exception("Matrices should be square")
    D= []
    for i in range(n):
        D = scipy.linalg.block_diag(D,diaA)

    B = np.kron(B,np.ones((n2,n2)))
    B[D.nonzero()] = 0
    A = B+D

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

    if mode == 'undirected' and not np.all(omega.T == omega):
        raise Exception("You should provide a symmetric affinity matrix")

    rowlist = []
    for i in xrange(ni):
        collist = []
        for j in xrange(nj):
            pij = omega[i,j]
            A1  = sample_from_block(nc[i],nc[j],pij)
            collist.append(A1)
        A1 = scipy.sparse.hstack(collist)
        rowlist.append(A1)

    A = scipy.sparse.vstack(rowlist)

    if mode == 'undirected':
        A = scipy.sparse.triu(A,k=1)
        A = A + A.T

    return A

def sample_from_block(m,n,pij):
    if pij == 0:
        block = scipy.sparse.csr_matrix((m,n))
    else:
        block = (scipy.sparse.random(m,n,density=pij) >0)*1
    return block
