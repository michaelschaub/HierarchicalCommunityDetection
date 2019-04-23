""" Script file to test various agglomeration mechanisms for hier clustering"""
from __future__ import division
import numpy as np
import scipy
import scipy.linalg
from scipy.stats import ortho_group
from scipy.signal import argrelmin

import GHRGbuild
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from helperfunctions import project_orthogonal_to, compute_number_links_between_groups, expand_partitions_to_full_graph

def find_subspace_angle_between_ev_bases(U,V):
    Q1 = U.T.dot(V)
    sigma = scipy.linalg.svd(Q1, compute_uv=False)
    ang = -np.ones(sigma.size)
    cos_index = sigma**2 <= 0.5
    if np.all(cos_index):
        ang= np.arccos(sigma)
    else:
        Q2 = V - U.dot(Q1)
        sigma2 = scipy.linalg.svd(Q2,compute_uv=False)
        ang[cos_index] = np.arccos(sigma[cos_index])
        sin_index = np.bitwise_not(cos_index)
        ang[sin_index] = np.arcsin(sigma2[sin_index])

    return ang, sigma

def project_to(subspace_basis, vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space to which we want to project
    vectors_to_project: project these vectors into the specified subspace
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
    X1 = S.T * V
    X2 = S.T * S
    projected = S * scipy.sparse.linalg.spsolve(X2, X1)

    return projected


def calculate_proj_error3(evecs, H, norm):
    n, k = np.shape(H)
    if n == k:
        error = 0
        return error
    V = evecs[:, :k]
    proj1 = project_orthogonal_to(H, V)

    if norm == 'F':
        error = scipy.linalg.norm(proj1)

    return error


def test_random_projection_with_kmeans2(n, k):
    nsamples = 20

    if n == k:
        return 0, 0

    error = np.zeros(nsamples)
    clust = KMeans(n_clusters=k)
    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:, :k]
        clust.fit(V)
        partition_vec = clust.labels_
        partition_vec = spectral.relabel_partition_vec(partition_vec)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        H = preprocessing.normalize(H, axis=0, norm='l2')
        error[i] = scipy.linalg.norm(project_orthogonal_to(H, V))

    return error


def hier_spectral_partition_agglomerate(A, partition, spectral_oper="Lap", model='SBM', reps=20, noise=1e-1, Ks=None, no_Ks_forward=False):
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
    2) While the list of of Ks is not empty, agglomerate network and try to identify partitions either using the identify_next_level function
    (unknown Ks, full list), or the identify_partitions_at_level function
    (known Ks, list of group sizes)
    3) these functions return a new Ks list, and a new partition to operate with
    according to which we agglomerate the network and start again from 1)
    """
    # TODO: the following options are not really made use / unclear effect
    # spectral_oper -- spectral operator used to perform clustering not used atm
    # model -- parameter for spectral clustering

    # k ==  number of groups == max index of partition +1
    k = np.max(partition)+1
    print "\n\nHIER SPECTRAL PARTITION -- agglomerative\n"
    print "Initial partition into", k, "groups \n"

    # Ks stores the candidate levels in inverse order
    # Note: set to min 1 group, as no agglomeration required
    # when only 2 groups are detected.
    if Ks is None:
        Ks = np.arange(1, k+1)
        find_levels = True
    else:
        find_levels = False

    # levels is a list of 'k' values of each level in the inferred hierarchy
    # pvec stores all hier. refined partitions
    levels = [k]
    pvec = []
    pvec.append(partition)
    list_candidate_agglomeration = Ks
    while len(Ks) > 0:

        print "List of partitions to assess: ", Ks, "\n"
        print "Current shape of network: ", A.shape, "\n"

        # TODO The normalization seems important for good results -- why?
        # should we normalize by varaiance of Bernoullis as well?
        Eagg, Nagg = compute_number_links_between_groups(A, partition)
        Aagg = Eagg / Nagg

        if find_levels:
            errors, std_errors, hier_partition_vecs = identify_next_level(
                Aagg, Ks, model=model, reg=False, norm='F', reps=reps, noise=noise)
            # kmax = np.max(Ks)
            selected = find_all_relevant_minima_from_errors(errors,std_errors,list_candidate_agglomeration)
            selected = selected -1
            selected = selected[:-1]
            Ks = Ks[selected]
            print "Minima at", Ks
            hier_partition_vecs = [hier_partition_vecs[si] for si in selected]
        else:
            k = Ks[0]
            print "ERROR"

        try:
            pvec.append(hier_partition_vecs[-1])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            # TODO: it might be useful to pass down candidates
            # from previous agglomeration rounds here instead of starting from scratch!
            if find_levels:
                if no_Ks_forward:
                    k = np.max(Ks)
                    Ks = np.arange(1, k+1)
                    list_candidate_agglomeration = Ks
                else:
                    list_candidate_agglomeration = Ks[selected]
                    k = np.max(Ks)
                    Ks = np.arange(1, k+1)
            levels.append(k)
            # if levels are not prespecified, reset candidates
            print 'partition into', k , ' groups'
            if k == 1:
                Ks = []

        # TODO: check below
        # this exception occurs when there is only a single candidate partition
        # and the error is not high enough
        # Check if  not better described as: if there is *no candidate* partition
        # (why only single candidate? why error not high enough -- low?!)
        except IndexError:
            pass

    print "HIER SPECTRAL PARTITION -- agglomerative\n Partitions into", levels, "groups \n"

    return pvec[::-1]
    # return expand_partitions_to_full_graph(pvec)[::-1]


def identify_next_level(A, Ks, model='SBM', reg=False, norm='F', reps=20, noise=1e-3):
    """
    Identify agglomeration levels by checking the projection errors and comparing the to a bootstrap
    verstion of the same network"""

    # first identify partitions and their projection error
    Ks, sum_errors, partition_vecs = identify_partitions_and_errors(A, Ks, model, reg, norm, partition_vecs=[])

    # repeat with noise
    if reps > 0:

        sum_errors = 0.
        std_errors = 0.
        m = 0.

        for _ in xrange(reps):
            Anew = spectral.add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(Anew, Ks, model, reg, norm, partition_vecs)
            sum_errors += errors

            # calculate online variance
            m_prev = m
            m = m + (errors - m) / (reps + 1)
            std_errors = std_errors + (errors - m) * (errors - m_prev)

        sum_errors /= reps
    std_errors = np.sqrt(std_errors/(reps-1))

    return sum_errors, std_errors, partition_vecs


def identify_partitions_and_errors(A, Ks, model='SBM', reg=False, norm='F', partition_vecs=[]):
    """
    Collect the partitions and projection errors found for a list Ks of 'putative' group numbers
    """

    max_k = np.max(Ks)

    L = spectral.construct_graph_Laplacian(A)
    Dtau_sqrt_inv = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        print L.shape, max_k
        ev, evecs = scipy.sparse.linalg.eigsh(L, max_k, which='SM', tol=1e-6)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:, index]

    # initialise errors
    error = np.zeros(len(Ks))
    # check if partitions are known
    partitions_unknown = partition_vecs == []

    # find partitions and their error for each k
    for ki, k in enumerate(Ks):
        if partitions_unknown:
            partition_vec, Hnorm = spectral.find_partition(evecs, k, model, Dtau_sqrt_inv, method='GMMspherical')
        else:
            partition_vec = partition_vecs[ki]
            Hnorm = spectral.create_normed_partition_matrix_from_vector(partition_vec, model)

        # calculate and store error
        error[ki] = calculate_proj_error3(evecs, Hnorm, norm)
        # print("K, error ")
        # print(k, error[ki])

        # save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs


def create_agglomerated_graphGHRG(groups_per_level=3,n_levels=3,n=3**10, snr=5):
    c_bar = 30
    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    G = D_actual.generateNetworkExactProb()
    A = D_actual.to_scipy_sparse_matrix(G)
    # get true hierarchy
    true_pvec = D_actual.get_partition_all()[-1]
    Eagg, Nagg = compute_number_links_between_groups(A, true_pvec)
    Aagg = Eagg / Nagg

    return Aagg, A, D_actual.get_partition_all()


def expected_errors_random_projection(dim_n,levels):
    """
    Compute vector of expected errors for random projection
        dim_n -- ambient space
        levels of hierarchy [1, ..., dim_n]
    """
    start_end_pairs = zip(levels[:-1],levels[1:])

    expected_error = []
    for i,j in start_end_pairs:
        Ks = np.arange(i,j)
        errors = np.sqrt((Ks - i) * (j - Ks) / dim_n)
        expected_error = np.hstack([expected_error,errors])
    expected_error = np.hstack([expected_error,0])
    return expected_error

def find_smallest_relevant_minima_from_errors(errors,std_errors,expected_error):
    #TODO: how to choose the threshold?
    relerror = np.zeros(expected_error.size)
    nonzero = expected_error != 0
    relerror[nonzero] = (errors[nonzero] + 2*std_errors[nonzero] - expected_error[nonzero]) / expected_error[nonzero]
    threshold = -0.6
    # relerror = (errors +2*std_errors) - expected_error
    # threshold = 0
    local_min = argrelmin(relerror)[0]
    below_thresh = np.nonzero(relerror < threshold)[0]
    levels = np.intersect1d(local_min, below_thresh).astype(int)
    Ks = np.arange(1, errors.size+1)
    print "Ks, local_min, below_thresh"
    print Ks, Ks[local_min], Ks[below_thresh]
    best_level = -1
    if levels.size > 0:
        best_level = levels[np.argmin(relerror[levels])]+1

    plt.figure(9)
    plt.plot(np.arange(1,errors.size+1), relerror)
    return best_level

def find_all_relevant_minima_from_errors(errors,std_errors,list_candidate_agglomeration):
    levels = [1,errors.size]
    expected_error = expected_errors_random_projection(errors.size,levels)
    next_level = find_smallest_relevant_minima_from_errors(errors,std_errors, expected_error)
    while next_level != -1:
        levels = levels + [next_level]
        levels.sort()
        expected_error = expected_errors_random_projection(errors.size, levels)
        next_level = find_smallest_relevant_minima_from_errors(errors, std_errors,expected_error)

    levels = np.intersect1d(np.array(levels),list_candidate_agglomeration)
    return np.array(levels)



def compare_agglomeration_variants():
    """ Compare various agglomeration techniques"""

    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    Aagg, Aorg, true_pvec = create_agglomerated_graphGHRG()
    pvec_inf = spectral.hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                            spectral_oper="Lap",
                                                            model='SBM',
                                                            reps=10,
                                                            noise=1e-3,
                                                            Ks=None)
    print "INFERRED PARTITION OLD, TRUE PARTITION"
    old_part = [len(np.unique(pv)) for pv in pvec_inf]
    true_part = [len(np.unique(pv)) for pv in true_pvec]
    print old_part
    print true_part
    if old_part == true_part:
        print "Nothing interesting happening" 

    plt.close('all')
    plt.figure()
    plt.imshow(Aagg-np.diag(np.diag(Aagg)))

    L = spectral.construct_graph_Laplacian(Aagg)
    ev, evecs = scipy.linalg.eigh(L)
    plt.figure()
    plt.plot(ev, 'x')


    # ############
    # Variant 2:
    # get raw projeciton errors and compare to analytical value in 'relative' manner
    print "\n\n START NEW TESTS"
    kmax, _ = Aagg.shape
    Ks = np.arange(1, kmax+1)
    errors, std_errors, partition_vecs = identify_next_level(Aagg, Ks)

    true_levels = [len(np.unique(pv)) for pv in true_pvec]
    true_levels = [1] + true_levels
    expected_error = expected_errors_random_projection(kmax,true_levels)
    candidates = expected_error - (errors + 2*std_errors) > 0
    plt.figure()
    plt.errorbar(Ks, errors, std_errors)
    plt.plot(Ks, expected_error)
    expected_error = np.sqrt((Ks - 1)*(kmax-Ks) / (kmax))
    plt.plot(Ks, expected_error)

    # set first and last error to 1 to avoid division by zero
    # these are not local minima in any case
    expected_error = np.sqrt((Ks - 1)*(kmax-Ks) / (kmax))
    levels = find_all_relevant_minima_from_errors(errors,std_errors,Ks)

    print "\n\nINFERRED LEVELS ONE-SHOT"
    print levels
    for i in (levels-1):
        print partition_vecs[i]

    relerror = np.zeros(expected_error.size)
    nonzero = expected_error != 0
    relerror[nonzero] = (errors[nonzero] - expected_error[nonzero]) / expected_error[nonzero]
    plt.figure()
    plt.plot(Ks, relerror)
    plt.figure()
    plt.plot(Ks, std_errors)

    # WITH INTERMEDIATE AGGREGATION
    pvec_inf = hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                   spectral_oper="Lap", model='SBM',
                                                   reps=10, noise=1e-3, Ks=None)

    print "INFERRED PARTITION2, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]
    for p in pvec_inf:
        print p

def compare_agglomeration_variants_no_hier():
    """ Compare various agglomeration techniques"""

    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    Aagg, Aorg, true_pvec = create_agglomerated_graphGHRG(groups_per_level=27,n_levels=1,n=3**10)
    pvec_inf = spectral.hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                            spectral_oper="Lap",
                                                            model='SBM',
                                                            reps=10,
                                                            noise=1e-3,
                                                            Ks=None)
    print "INFERRED PARTITION OLD, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]

    plt.close('all')
    plt.figure()
    plt.imshow(Aagg-np.diag(np.diag(Aagg)))

    L = spectral.construct_graph_Laplacian(Aagg)
    ev, evecs = scipy.linalg.eigh(L)
    plt.figure()
    plt.plot(ev, 'x')


    # ############
    # Variant 2:
    # get raw projeciton errors and compare to analytical value in 'relative' manner
    print "\n\n START NEW TESTS"
    kmax, _ = Aagg.shape
    Ks = np.arange(1, kmax+1)
    errors, std_errors, partition_vecs = identify_next_level(Aagg, Ks)

    true_levels = [len(np.unique(pv)) for pv in true_pvec]
    true_levels = [1] + true_levels
    expected_error = expected_errors_random_projection(kmax,true_levels)
    candidates = expected_error - (errors + 2*std_errors) > 0
    plt.figure()
    plt.errorbar(Ks, errors, std_errors)
    plt.plot(Ks, expected_error)
    expected_error = np.sqrt((Ks - 1)*(kmax-Ks) / (kmax))
    plt.plot(Ks, expected_error)

    # set first and last error to 1 to avoid division by zero
    # these are not local minima in any case
    expected_error = np.sqrt((Ks - 1)*(kmax-Ks) / (kmax))
    levels = find_all_relevant_minima_from_errors(errors,std_errors,Ks)

    print "\n\nINFERRED LEVELS ONE-SHOT"
    print levels
    for i in (levels-1):
        print partition_vecs[i]

    relerror = np.zeros(expected_error.size)
    nonzero = expected_error != 0
    relerror[nonzero] = (errors[nonzero]) / expected_error[nonzero]
    plt.figure()
    plt.plot(Ks, relerror)
    plt.figure()
    plt.plot(Ks, std_errors)

    # WITH INTERMEDIATE AGGREGATION
    pvec_inf = hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                   spectral_oper="Lap", model='SBM',
                                                   reps=10, noise=1e-3, Ks=None)

    print "INFERRED PARTITION2, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]
    for p in pvec_inf:
        print p