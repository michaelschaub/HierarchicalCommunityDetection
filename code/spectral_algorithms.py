#!/usr/bin/env python
from __future__ import division
import scipy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as linalg
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import mixture
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.signal import argrelextrema
import scipy.linalg
import scipy.stats
from scipy.signal import argrelextrema
from sys import stdout

# GLOBAL TODO / CHECKLIST
# mode parameter in identify_next_level -- SBM or DCSBM?


def hier_spectral_partition(A,method_agg='Lap',method_zoom='Bethe',first_pass='Bethe', reps=10, noise=1e-3):

    # initial spectral clustering, performed according to method 'first pass'
    p0 = spectral_partition(A,mode=first_pass,num_groups=-1)

    # agglomerate builds list of all partitions
    pvec_agg = hier_spectral_partition_agglomerate(A,p0, mode=method_agg, reps=reps, noise=noise)
    

    return pvec_agg


def hier_spectral_partition_agglomerate(A, partition, mode="Lap", reps=10, noise=1e-3):
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
    Ks=np.arange(k,1,-1)
    
    # levels is a list of 'k' values of each level in the inferred hierarchy
    levels = [k+1]
    while len(Ks)>0:

        Eagg, Nagg = compute_number_links_between_groups(A,partition)
        Aagg = Eagg / Nagg
        Ks, hier_partition_vecs = identify_next_level(Aagg,max_k=k,mode='SBM',reg=False, norm='F', threshold=1/3, reps=reps, noise=noise)

        try:
            pvec.append(hier_partition_vecs[0])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            k=Ks[0]-1
            levels.append(k+1)
            print 'partition into', k+1 ,' groups'
            if k==1:
                Ks=[]

        #TODO: check if below is not better described as: if there is *no candidate* partition (why only single candidate? why error not high enough -- low?!)
        #this exception occurs when there is only a single candidate partition and the error is not high enough.
        except IndexError:
            pass


    print "HIER SPECTRAL PARTITION -- agglomerative\n Partitions into", levels, "groups \n"

    return expand_partitions_to_full_graph(pvec)[::-1]



def spectral_partition(A, mode='Lap', num_groups=2, regularizer='BHa'):
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
        partition = cluster_with_BetheHessian(A,num_groups=num_groups,mode='unweighted', regularizer=regularizer)

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

def regularized_laplacian_spectral_clustering(A, num_groups=2,
                                              tau=-1,clustermode='kmeans'):
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

    if clustermode == 'kmeans':
        X = preprocessing.normalize(evecs, axis=1, norm='l2')

        clust = KMeans(n_clusters = num_groups)
        clust.fit(X)
        partition_vector = clust.labels_
    elif clustermode == 'qr':
        #TODO: normalize EV?
        partition_vector = clusterEVwithQR(evecs)


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
        ev_pos, evecs_pos = scipy.sparse.linalg.eigsh(BH_pos,num_groups,which='SA')
        ev_neg, evecs_neg = scipy.sparse.linalg.eigsh(BH_neg,num_groups,which='SA')
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

##########################################
# X LAPLACIAN
##########################################
def cluster_with_XLaplacian(A, number_groups, learning_rate=5):
    X = 0
    LX = A + X
    thres = 1/A.shape[0]
    has_converged = False

    while has_converged:
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


def inverse_participation_ratio(vec):
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
        # print "K, error: "
        # print k, e
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


def identify_next_level(A,max_k=-1,mode='SBM',reg=False, norm='F', threshold=1/3, reps=10, noise=1e-3):

    #determine set of candidate k's
    if max_k == -1:
        max_k = A_.shape[0]
    Ks=np.arange(max_k,1,-1)

    #first identify partitions and their projection error
    Ks, sum_errors, partition_vecs = identify_partitions_and_errors(A,Ks,mode,reg, norm,partition_vecs=[])

    #repeat with noise
    if reps>0:

        sum_errors = 0
        std_errors = 0.
        m = 0.

        for rep in xrange(reps):
            Anew = add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(Anew,Ks,mode,reg, norm,partition_vecs)
            sum_errors+=errors

            #calculate online variance
            m_prev = m
            m = m + (errors - m) / (reps+1)
            std_errors = std_errors + (errors - m) * (errors - m_prev)


        sum_errors/=reps

    std_errors=np.sqrt(std_errors)
    #find errors below threshold 
    below_thresh = (sum_errors<threshold) 
    
    
    levels = find_local_minima(sum_errors)
    print below_thresh.nonzero()[0]
    print 'sum_errors',zip(Ks[levels],sum_errors[levels])
    print 'sum_errors',zip(Ks[below_thresh],sum_errors[below_thresh])
    #choose only local minima that are below threshold
    levels = np.intersect1d(levels,below_thresh.nonzero()[0])
    print 'Levels inferred=',len(levels), Ks[levels], sum_errors[levels], std_errors[levels]
    hier_partition_vecs=[partition_vecs[si] for si in levels]
    return Ks[levels], hier_partition_vecs





def identify_partitions_and_errors(A,Ks,mode='SBM',reg=False, norm='F',partition_vecs=[]):
    max_k = Ks[0]
    
    L, Dtau_sqrt_inv = construct_normalised_Laplacian(A, reg)
    if reg:
        # set tau to average degree
        tau = A.sum()/A.shape[0]
    else:
        tau = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    sparse_input = False
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        print L.shape, max_k
        ev, evecs = scipy.sparse.linalg.eigsh(L,Ks[0],which='LM')
        sparse_input = True
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]

    #initialise errors
    error = np.zeros(len(Ks))
    #check if partitions are known
    partitions_unknown= partition_vecs==[]

    #find partitions and their error for each k
    for ki,k in enumerate(Ks):
        if partitions_unknown:
            partition_vec, Hnorm = find_partition(evecs, k, tau, norm, mode, Dtau_sqrt_inv)
        else :
            partition_vec = partition_vecs[ki]
            Hnorm = create_normed_partition_matrix_from_vector(partition_vec,mode)
        
        #calculate and store error
        error[ki] = calculate_proj_error(evecs, Hnorm, norm)
        #~ print("K, error ")
        #~ print(k, error[ki])

        #save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs



def find_partition(evecs, k, tau, norm, mode, Dtau_sqrt_inv, method='QR', n_init=20):
    V = evecs[:,:k]
    if mode == 'DCSBM':
        X = preprocessing.normalize(V, axis=1, norm='l2')

    elif mode == 'SBM':
        X = Dtau_sqrt_inv* V

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
    H = create_normed_partition_matrix_from_vector(partition_vec,mode)

    return partition_vec, H


def calculate_proj_error(evecs,H,norm):
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

def construct_normalised_Laplacian(Omega, reg):
    if reg:
        # set tau to average degree
        tau = Omega.sum()/Omega.shape[0]
    else:
        tau = 0

    # construct normalised Laplacian
    Dtau_sqrt_inv = scipy.sparse.diags(np.power(np.array(Omega.sum(1)).flatten() + tau,-.5),0)
    # print Omega
    L = Dtau_sqrt_inv.dot(Omega)
    L = Dtau_sqrt_inv.dot(L.T).T
    L = (L+L.T)/2

    return L, Dtau_sqrt_inv



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

##################################################
# QR Decomposition for finding clusters
##################################################

def clusterEVwithQR(EV, randomized=False, gamma=4):
    """Given a set of eigenvectors find the clusters of the SBM"""
    if randomized is True:
        Z, Q = orthogonalizeQR_randomized(EV,gamma)
    else:
        Z, Q = orthogonalizeQR(EV)

    cluster_ = scipy.absolute(Z).argmax(axis=1)

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
    elements = np.random.choice(elements, count, p=probabilities)   #changed to np.random (since scipy.random is not a real module)
    ellemens = scipy.unique(elements)


    Q, R, P = scipy.linalg.qr(EV[elements,:].T, mode='economic', pivoting=True)
    # get indices of k representative points
    P = P[:k]

    # polar decomposition to find nearest orthogonal matrix
    U, S, V = scipy.linalg.svd(EV[P,:].T,full_matrices=False)

    Z = EV.dot(U.dot(V.T))

    return Z, Q

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

def create_normed_partition_matrix_from_vector(partition_vec,mode):

    H = create_partition_matrix_from_vector(partition_vec)

    if mode == 'DCSBM':
        Dsqrt = scipy.sparse.diags(scipy.sqrt(Omega.sum(axis=1)+tau).flatten())
        H = Dtau_sqrt.dot(H)

    # normalize column norm to 1 of the partition indicator matrices
    return preprocessing.normalize(H,axis=0,norm='l2')


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


def compute_number_links_between_groups(A,partition_vec,directed=True):
    """
    Compute the number of possible and actual links between the groups indicated in the
    partition vector.
    TODO: option to declare whether self-loops should be accounted for!?
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

    if not directed:
        links_between_groups = links_between_groups - np.diag(np.diag(links_between_groups))/2.0
        links_between_groups = np.triu(links_between_groups)

    # convert to array type first, before performing outer product
    nodes_per_group = pmatrix.sum(0).A
    possible_links_between_groups = np.outer(nodes_per_group,nodes_per_group)

    if not directed:
        possible_links_between_groups = possible_links_between_groups - np.diag(nodes_per_group.flatten())
        possible_links_between_groups = possible_links_between_groups - np.diag(np.diag(possible_links_between_groups))/2.0
        possible_links_between_groups = np.triu(possible_links_between_groups)


    return links_between_groups, possible_links_between_groups

def expand_partitions_to_full_graph(pvecs):
    """
    Map aggregated partition vectors to full-sized partition vectors
    """

    # partiitions are stored relative to the size of the aggregated graph, so we have to
    # expand them again into the size of the full graph

    # the fines partition is already at the required size
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

def find_local_minima(vec):
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

"""Add some small random noise to a (dense) small matrix as a perturbation"""
def add_noise_to_small_matrix(M,snr=0.001,noise_type="gaussian"):

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

def xlogy(x,y):
    """ compute x log(y) elementwise, with the convention that 0log0 = 0"""
    xlogy = x*np.log(y)
    xlogy[np.isinf(xlogy)] = 0
    xlogy[np.isnan(xlogy)] = 0
    return xlogy

def compute_likelihood_SBM(pvec,A,omega=None):
    H = create_partition_matrix_from_vector(pvec)
    # self-loops and directedness is not allowed here
    Emat, Nmat = compute_number_links_between_groups(A,pvec,directed=False)
    if omega is None:
        omega = Emat / Nmat

    logPmat = xlogy(Emat,omega) + xlogy(Nmat-Emat,1 - omega)
    likelihood = logPmat.sum()
    return likelihood
