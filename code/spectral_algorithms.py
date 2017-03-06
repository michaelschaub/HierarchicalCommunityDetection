#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as linalg
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.signal import argrelextrema

def hier_spectral_partition(A,ma='BH_plus_Spectral',mz='Bethe'):
    pvec_agg = hier_spectral_partition_agglomerate(A,mode=ma)
    pvec_agg = expand_partitions_to_full_graph(pvec_agg)
    p0 = pvec_agg[0]

    pvec_zoom = hier_spectral_partition_zoom_in(A,p0,mode=mz)

    pvec_tot = pvec_zoom + pvec_agg

    return pvec_tot


def hier_spectral_partition_zoom_in(A, partition, mode='Bethe', zgroups = -1):
    nc = np.max(partition)+1

    pzoom = []
    keep_looping = True
    print "RECURSIVE SPLITTING"
    while keep_looping:
        max_k = 0
        pvec = np.zeros_like(partition)

        for ii in xrange(nc):
            Anew = A[:,partition==ii]
            Anew = Anew[partition==ii,:]

            pnew = spectral_partition(Anew,mode,zgroups)
            if pvec.max() == 0:
                pvec[partition==ii] = pnew + max_k
                max_k = pvec.max()
            else:
                pvec[partition==ii] = pnew + max_k + 1
                max_k = pvec.max()

        if pvec.max() != 0:
            print "RECURSIVE SPLIT FOUND", pvec.max()+1, "groups"
            pzoom.append(pvec)
            partition = pvec
        else:
            keep_looping = False

    return pzoom


def hier_spectral_partition_agglomerate(A, mode="BH_plus_Spectral"):
    """Performs spectral clustering and hierarchical agglomeration based on the provided mode parameter"""

    if mode == "BH_plus_Spectral":
        first_round_method = 'Bethe'
        first_round_num_groups = -1
        method_agglomeration = "Lap"
        model_select_agglomeration = 'spectral'

    elif mode == "LS_plus_Spectral":
        first_round_method = 'SeidelLap'
        first_round_num_groups = -1
        method_agglomeration = "Lap"
        model_select_agglomeration = 'spectral'

    #FIRST RUN --- try to partition the graph
    pvec = []
    partition = spectral_partition(A,mode=first_round_method,num_groups=first_round_num_groups)

    #STORE RESULTS
    pvec.append(partition)
    k = np.max(partition)+1
    k0 =k
    print "HIER SPECTRAL PARITION -- agglomerative\n Initial partition into", k0, "groups \n"
    while k > 1:
        links_between_groups, possible_links_between_groups = compute_number_links_between_groups(A,partition)
        A = links_between_groups
        # print "Aggregated network:"
        # print links_between_groups
        if method_agglomeration == "Lap":
            k, partition, H, error = identify_hierarchy_in_affinity_matrix(links_between_groups)
        else:
            pass
        if partition is not None:
            pvec.append(partition)


    return pvec


def spectral_partition(A, mode='Lap', num_groups=2):
    """ Perform one round of spectral clustering for a given network matrix A
    Inputs: A -- input adjacency matrix
            mode -- variant of spectral clustering to use (Laplacian, Bethe Hessian,
            Non-Backtracking, XLaplacian, ...)
            num_groups -- in how many groups do we want to split the graph?
            (default: 2; set to -1 to infer number of groups from spectrum)

            Output: partition_vec -- clustering of the nodes
    """

    if   mode == "Lap":
        if num_groups != -1:
            partition, _ = regularized_laplacian_spectral_clustering(A,num_groups=num_groups)

    elif mode == "Bethe":
        partition = cluster_with_BetheHessian(A,num_groups=num_groups,mode='weighted')

    elif mode == "NonBack":
        pass

    elif mode == "XLaplacian":
        pass

    elif mode == "SeidelLap":
        if num_groups != -1:
            partition, _ = cluster_with_SLaplacian_simple(A,num_groups=num_groups)
        else:
            k, partition, _, __ = cluster_with_SLaplacian_and_model_select(A,num_groups=num_groups)

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

    A = test_sparse_and_transform(A)

    # check if tau regularisation parameter is specified otherwise go for mean degree...
    if tau==-1:
        # set tau to average degree
        tau = A.sum()/A.shape[0]

    d = np.array(A.sum(axis=1)).flatten().astype(float)
    Dtau_sqrt_inv = scipy.sparse.diags(np.power(d + tau,-.5),0)
    L = Dtau_sqrt_inv.dot(A).dot(Dtau_sqrt_inv)


    # compute eigenvalues and eigenvectors (sorted according to magnitude first)
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
    rA = scipy.sparse.csr_matrix((rA_data,A.nonzero()))

    # full Bethe Hessian
    BHw = DD - rA
    return BHw


def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa', mode='weighted'):
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
        relevant_ev, _ = find_negative_eigenvectors(BH_pos)
        X = relevant_ev

        relevant_ev, _ = find_negative_eigenvectors(BH_neg)
        X = np.hstack([X, relevant_ev])
        num_groups = X.shape[1]

        if num_groups == 0:
            print "no indication for grouping -- return all in one partition"
            partition_vector = np.zeros(A.shape[0],dtype='int')
            return partition_vector

    else:
        # TODO: note that we combine the eigenvectors of pos/negative BH and do not use
        # information about positive / negative assortativity here
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
# X LAPLACIAN
##########################################
def cluster_with_XLaplacian(A, number_groups, learning_rate=5):
    X = 0
    LX = A + X
    thres = 1/A.shape[0]
    has_converged = False

    while has_converged
        Iratio_max = -999
        index_max = -1
        ev, evecs = scipy.sparse.linalg.eigsh(LX,number_groups,'LA')
        for i in np.arange(number_groups):
            Iratio = inverse_participation_ratio(evecs[:,i])
            if Iratio > Iratio_max:
                Iratio_max = Iratio
                index_max = i

        if Iratio_max < thres:
            has_converged = True

        else:
            X = X - learning_rate*scipy.sparse.diags(np.power(evecs[:,index_max],2))
            LX = A+X

    clust = KMeans(n_clusters = num_groups)
    clust.fit(evecs)
    partition_vector = clust.labels_

    return partition_vector, evecs


def inverse_participation_ratio(vec)
    return np.power(vec,4).sum()


##########################################
# SEIDEL LAPLACIAN
##########################################
def create_seidel_lap_operator(A,rho=None):
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)

    n = A.shape[0]
    I = scipy.sparse.diags(np.ones(n),0)
    d = A.sum(axis=1).A.flatten().astype(float)
    if rho==None:
        rho = d.mean()/n
    dtot = d*(1-rho) + rho * (n-1)
    Ds_invs = scipy.sparse.diags(np.power(dtot,-0.5),0)

    def seidel_lap_mat_vec(x):
        mv = x - (1+rho)*Ds_invs*A*Ds_invs*x - rho*Ds_invs*Ds_invs*x
        mv += Ds_invs*scipy.ones(n)*(scipy.ones(n)*Ds_invs*x)
        return mv

    LS = LinearOperator((n,n),matvec=seidel_lap_mat_vec)
    return LS, Ds_invs

def cluster_with_SLaplacian_simple(A,num_groups,rho=None):
    # compute eigenvalues and eigenvectors (sorted according to smallest)
    L, _ = create_seidel_lap_operator(A)
    ev, evecs = scipy.sparse.linalg.eigsh(L,num_groups,which='SA')

    clust = KMeans(n_clusters = num_groups)
    clust.fit(evecs)
    partition_vector = clust.labels_

    return partition_vector, evecs

def cluster_with_SLaplacian_and_model_select(A,num_groups,rho=None,max_k=16,mode='SBM'):
    # compute eigenvalues and eigenvectors (sorted according to smallest)
    L, Ds_invs = create_seidel_lap_operator(A)
    ev, evecs = scipy.sparse.linalg.eigsh(L,max_k,which='SA')
    print ev

    print "START MODEL SELECTION PHASE"

    n = L.shape[0]
    error = np.zeros(max_k)

    #TODO: check all these cases carefully!
    for k in xrange(1,max_k):
        if mode == 'DCSBM':
            error("NOT FULLY DEVELOPED YET!!")
            pass

        elif mode == 'SBM':
            V = evecs[:,:k]
            # print "V"
            # print V, V.shape
            X = preprocessing.normalize(V, axis=1, norm='l2')
            clust = KMeans(n_clusters = k)
            clust.fit(X)
            partition_vec = clust.labels_
            partition_vec = relabel_partition_vec(partition_vec)
            # print partition_vec

            H = create_partition_matrix_from_vector(partition_vec)
            H = Ds_invs.dot(H)
            H = preprocessing.normalize(H,axis=0,norm='l2')
        else:
            error('something went wrong. Please specify valid mode')

        proj1 = project_orthogonal_to(H,V)
        proj2 = project_orthogonal_to(V,H)
        norm1 = scipy.linalg.norm(proj1)
        norm2 = scipy.linalg.norm(proj2)
        # print norm1, norm2
        e = 0.5*(norm1+norm2)
        print "K, error: "
        print k, e
        error[k]=e

    local_min = argrelextrema(error,np.less)
    print local_min[-1]
    if local_min is None:
        return 1, None, None, None
    else:
        kbest = local_min[-1][-1]
        V = evecs[:,:kbest]
        X = preprocessing.normalize(V, axis=1, norm='l2')
        clust = KMeans(n_clusters = kbest)
        clust.fit(X)
        partition_vec = clust.labels_
        partition_vec = relabel_partition_vec(partition_vec)
        H = create_partition_matrix_from_vector(partition_vec)

        return kbest, partition_vec, H, error[kbest]


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

##################################################
# SPECTRAL MODEL SELECTION VIA INVARIANT SUBSPACE
##################################################

def identify_hierarchy_in_affinity_matrix(Omega,mode='SBM',reg=False):
    #TODO: atm this uses just the standard laplacian which should concentrate as we are
    # in the aggregated regime

    max_k = Omega.shape[0]
    #TODO: we might want to adjust this later on
    thres = 0.02

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

    #NOTE: THIS is not a Laplacian but the normalized adjacency of Rohe et al..
    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    print "START AGGLOMERATION PHASE"
    # print "evec_sorted"
    # print evecs

    #TODO: check all these cases carefully!
    for k in xrange(max_k-1,1,-1):
        if mode == 'DCSBM':
            V = evecs[:,:k]
            # print "V"
            # print V
            X = preprocessing.normalize(V, axis=1, norm='l2')
            clust = KMeans(n_clusters = k)
            clust.fit(X)
            partition_vec = clust.labels_
            partition_vec = relabel_partition_vec(partition_vec)
            # print partition_vec

            H = create_partition_matrix_from_vector(partition_vec)
            Dsqrt = scipy.sparse.diags(scipy.sqrt(Omega.sum(axis=1)+tau).flatten())
            H = Dtau_sqrt.dot(H)
            H = preprocessing.normalize(H,axis=0,norm='l2')

        elif mode == 'SBM':
            V = evecs[:,:k]
            # print "V"
            # print V
            X = preprocessing.normalize(V, axis=1, norm='l2')
            clust = KMeans(n_clusters = k)
            clust.fit(X)
            partition_vec = clust.labels_
            partition_vec = relabel_partition_vec(partition_vec)
            # print partition_vec

            H = create_partition_matrix_from_vector(partition_vec)
            H = Dtau_sqrt_inv.dot(H)
            H = preprocessing.normalize(H,axis=0,norm='l2')
        else:
            error('something went wrong. Please specify valid mode')

        proj1 = project_orthogonal_to(H,V)
        proj2 = project_orthogonal_to(V,H)
        norm1 = scipy.linalg.norm(proj1)
        norm2 = scipy.linalg.norm(proj2)
        error = 0.5*(norm1+norm2)
        # print "K, error: "
        # print k, error
        # Note that this should always be fulfilled at k=1
        if error < thres*max_k:
            print "Agglomerated into " + str(k) + " groups \n\n"
            return k, partition_vec, H, error

    #TEST if there are indications for final/global agglomeration
    # equitable condition
    # actually this last part might be unnecessary..
    AH = Omega.sum(axis=1) / np.sqrt(max_k)
    HHpAH = np.ones_like(AH)*AH.mean()
    error = scipy.linalg.norm(AH - HHpAH)
    if error < thres*max_k:
        partition_vec = np.zeros((1,max_k))
        H = create_partition_matrix_from_vector(partition_vec)
        k = 1
        print "Final agglomeration: yes \n"
        return 1, partition_vec, H, error
    else:
        partition_vec = None
        H = -1
        k = 0
        print "Final agglomeration: no \n"
        return k , partition_vec, H, error



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

    # compute S*(S^T*S)^{-1}*S'*V
    projected = S*scipy.sparse.linalg.spsolve(S.T*S,S.T*V)

    orthogonal_proj = V - projected
    return orthogonal_proj



#######################################################
# HELPER FUNCTIONS
#######################################################
def relabel_partition_vec(pvec):
    k = pvec.max()+1
    if k ==1:
        return pvec
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



def test_sparse_and_transform(A):
    """ Check if matrix is sparse and if not, return it as sparse matrix"""
    if not scipy.sparse.issparse(A):
        print "Input matrix not in sparse format, transforming to sparse matrix"
        A = scipy.sparse.csr_matrix(A)
    return A


def compute_number_links_between_groups(A,partition_vec):
    """
    Compute the number of possible and actual links between the groups indicated in the
    partition vector.
    """

    pmatrix = create_partition_matrix_from_vector(partition_vec)
    # number of columns is number of groups
    nr_groups = pmatrix.shape[1]

    if not scipy.sparse.issparse(A):
        A = scipy.mat(A)

    # all inputs are matrices here -- calculation works accordingly and transforms to
    # array only afterwards
    # each block counts the number of half links / directed links
    links_between_groups = pmatrix.T * A * pmatrix
    links_between_groups = links_between_groups.A

    # convert to array type first, before performing outer product
    nodes_per_group = pmatrix.sum(0).A
    possible_links_between_groups = np.outer(nodes_per_group,nodes_per_group)


    return links_between_groups, possible_links_between_groups

def expand_partitions_to_full_graph(pvecs):
    pvec_new = []
    pvec_new.append(pvecs[0])

    for i in xrange(len(pvecs)-1):
        pold = pvecs[i]
        pnew = pvecs[i+1]
        partition = pnew[pold]
        pvec_new.append(partition)

    return pvec_new
