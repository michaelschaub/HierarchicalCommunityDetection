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
        Ks -- list of the partition sizes at each level 
              (e.g. [3, 9, 27] for a hierarchical split into 3 x 3 x 3 groups.

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
        p0 = spectral.spectral_partition(A, spectral_oper=first_pass, num_groups=-1)
        Ks = []
        Ks.append(np.max(p0) + 1)


    # SECOND STEP
    # initial spectral clustering using spectral_oper; Lap == Rohe Laplacian
    p0 = spectral.spectral_partition(A, spectral_oper=spectral_oper, num_groups=Ks[-1])

    # if Ks only specifies number of clusters at finest level,
    # set again to none for agglomeration
    if len(Ks) == 1:
        Ks = None
    else:
        Ks = Ks[:-1]

    # THIRD STEP
    # Now we build a list of all partitions
    pvec_agg = hier_spectral_partition_agglomerate(
        A, p0, spectral_oper=spectral_oper, model=model, reps=reps, noise=noise, Ks=Ks)

    return pvec_agg

def hier_spectral_partition_agglomerate(A, partition, spectral_oper="Lap", model='SBM', reps=20, noise=2e-2, Ks=None, ind_levels_across_agg=False):
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

        print "\n\nList of partitions to assess: ", Ks, "\n"
        print "Current shape of network: ", A.shape, "\n"
        print "Current levels: ", levels, "\n"

        # TODO The normalization seems important for good results -- why?
        # should we normalize by varaiance of Bernoullis as well?
        Eagg, Nagg = compute_number_links_between_groups(A, partition)
        Aagg = Eagg / Nagg

        if find_levels:
            errors, std_errors, hier_partition_vecs = identify_next_level(
                Aagg, Ks, model=model, reg=False, norm='F', reps=reps, noise=noise)
            plt.figure(125)
            plt.errorbar(Ks, errors,std_errors)
            # kmax = np.max(Ks)
            selected, below_thresh = find_all_relevant_minima_from_errors(errors,std_errors,list_candidate_agglomeration)
            selected = selected -1
            print "selected (before), ", selected
            selected = selected[:-1]
            print "Ks: ", Ks, " selected: ", selected
            Ks = Ks[selected]
            print "Minima at", Ks
            hier_partition_vecs = [hier_partition_vecs[si] for si in selected]
        else:
            k = Ks[-1]
            Ks, hier_partition_vecs = identify_partitions_at_level(
                Aagg, Ks, model=model, reg=False)

        try:
            pvec.append(hier_partition_vecs[-1])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            if find_levels:
                if ind_levels_across_agg:
                    k = np.max(Ks)
                    Ks = np.arange(1, k+1)
                    list_candidate_agglomeration = Ks
                else:
                # TODO: it might be useful to pass down candidates
                # from previous agglomeration rounds here instead of starting from scratch!
                    list_candidate_agglomeration = below_thresh + 1
                    print "updated list"
                    print list_candidate_agglomeration
                    k = np.max(Ks)
                    Ks = np.arange(1, k+1)

            levels.append(k)
            # if levels are not prespecified, reset candidates
            print 'partition into', k , ' groups'
            if k == 1:
                Ks = []

        # this exception occurs *no candidate* partition (selected == [])
        # and indicates that agglomeration has stopped
        except IndexError:
            print "selectecd == [] ", selected
            pass

    print "HIER SPECTRAL PARTITION -- agglomerative\n Partitions into", levels, "groups \n"

    # return pvec[::-1]
    return expand_partitions_to_full_graph(pvec)[::-1]

def identify_partitions_at_level(A, Ks, model='SBM', reg=False):
    """
    For a given graph with (weighted) adjacency matrix A and list of
    partition sizes to assess (Ks), find the partition of a given size Ks[0]
    via the find_partition function using the model and regularization
    provided.
    Inputs:
        A -- (agglomerated) adjacency matrix of network
        Ks -- list of the partition sizes at each level 
              (e.g. [3, 9, 27] for a hierarchical split into 3 x 3 x 3 groups.
        model -- parameter for spectral clustering
        reg -- use regularization for spectral clustering?

    Outputs:
        Ks -- remaining list of group sizes to consider
        partition_vec -- found partition at this level
    """

    # L, Dtau_sqrt_inv, tau = construct_normalised_Laplacian(A, reg)
    L = spectral.construct_graph_Laplacian(A)
    Dtau_sqrt_inv = 0

    # get eigenvectors
    # input A may be a sparse scipy matrix or dense format numpy 2d array.
    try:
        ev, evecs = scipy.linalg.eigh(L)
    except ValueError:
        ev, evecs = scipy.sparse.linalg.eigsh(L, Ks[-1], which='SM', tol=1e-6)

    index = np.argsort(np.abs(ev))
    evecs = evecs[:, index]

    # find the partition for Ks[-1] groups
    partition_vec, Hnorm = spectral.find_partition(evecs, Ks[-1], model, Dtau_sqrt_inv)

    return Ks[:-1], [partition_vec]

def identify_next_level(A, Ks, model='SBM', reg=False, norm='F', reps=20, noise=1e-2):
    """
    Identify agglomeration levels by checking the projection errors and comparing the to a bootstrap
    verstion of the same network"""

    # first identify partitions and their projection error
    Ks, sum_errors2, partition_vecs = identify_partitions_and_errors(A, Ks, model, reg, norm, partition_vecs=[])

    plt.figure(125)
    plt.plot(Ks, sum_errors2, 'o')

    # repeat with noise
    if reps > 0:

        sum_errors = 0.
        std_errors = 0.
        m = 0.

        for kk in xrange(reps):
            Anew = spectral.add_noise_to_small_matrix(A, snr=noise)
            _, errors, _ = identify_partitions_and_errors(Anew, Ks, model, reg, norm, partition_vecs)
            sum_errors += errors

            # calculate online variance
            m_prev = m
            m = m + (errors - m) / (kk + 1)
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

        # save partition
        if partitions_unknown:
            partition_vecs.append(partition_vec)

    return Ks, error, partition_vecs

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
    ratio_error = np.zeros(expected_error.size) 

    # find points below relative error
    nonzero = expected_error != 0
    relerror[nonzero] = (errors[nonzero] + 3*std_errors[nonzero] - expected_error[nonzero]) / expected_error[nonzero]
    threshold = -0.5
    below_thresh = np.nonzero(relerror < threshold)[0]

    # find relative minima
    ratio_error[nonzero] = (errors[nonzero] + 3*std_errors[nonzero])/ expected_error[nonzero]
    local_min = argrelmin(ratio_error)[0]
    plt.figure(12)
    plt.plot(np.arange(1,ratio_error.size+1),ratio_error)

    # levels == below thres && local min
    levels = np.intersect1d(local_min, below_thresh).astype(int)

    # remove already found local minima from list
    Ks = np.arange(errors.size)
    levels = np.intersect1d(levels, Ks[nonzero]).astype(int)

    Ks = np.arange(1, errors.size+1)
    print "Ks, local_min, below_thresh, levels"
    print Ks, Ks[local_min], Ks[below_thresh], Ks[levels]
    best_level = -1
    if levels.size > 0:
        best_level = levels[np.argmin(relerror[levels])]+1
        print "agglomeration level candidate"
        print best_level

    # plt.figure(9)
    # plt.plot(np.arange(1,errors.size+1), relerror)
    return best_level, below_thresh

def find_all_relevant_minima_from_errors(errors,std_errors,list_candidate_agglomeration):
    """Given a set of error and standard deviations, find best minima"""
    levels = [1,errors.size]
    expected_error = expected_errors_random_projection(errors.size,levels)

    # plot candidate levels vs. errors
    plt.figure(222)
    Ks = np.arange(expected_error.size) + 1
    plt.plot(Ks,expected_error)
    plt.errorbar(Ks,errors,std_errors)

    next_level, below_thresh = find_smallest_relevant_minima_from_errors(errors,std_errors, expected_error)
    while next_level != -1:
        levels = levels + [next_level]
        levels.sort()
        expected_error = expected_errors_random_projection(errors.size, levels)
        next_level, below_thresh2 = find_smallest_relevant_minima_from_errors(errors, std_errors,expected_error)

    print "list_candidate_agglomeration"
    print list_candidate_agglomeration
    print "levels"
    print levels
    levels = np.intersect1d(np.array(levels),list_candidate_agglomeration)
    print "levels updated"
    print levels

    # remove again the level 1 entry
    if levels[0] == 1:
        levels = levels[1:]
    print "levels returned"
    print np.array(levels)

    return np.array(levels), below_thresh

def create_agglomerated_graphGHRG(groups_per_level=3,n_levels=3,n=3**9, snr=6.5):
    c_bar = 30
    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    G = D_actual.generateNetworkExactProb()
    A = D_actual.to_scipy_sparse_matrix(G)
    # get true hierarchy
    true_pvec = D_actual.get_partition_all()[-1]
    Eagg, Nagg = compute_number_links_between_groups(A, true_pvec)
    Aagg = Eagg / Nagg

    return Aagg, A, D_actual.get_partition_all()




def compare_agglomeration_variants():
    """ Compare various agglomeration techniques"""

    plt.close("all")
    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    Aagg, Aorg, true_pvec = create_agglomerated_graphGHRG(snr=3.5)
    pvec_inf = spectral.hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                            spectral_oper="Lap",
                                                            model='SBM',
                                                            reps=10,
                                                            noise=1e-2,
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
    print "\n\n START NEW TESTS ONE-SHOT"
    kmax, _ = Aagg.shape
    Ks = np.arange(1, kmax+1)
    errors, std_errors, partition_vecs = identify_next_level(Aagg, Ks)

    true_levels = [len(np.unique(pv)) for pv in true_pvec]
    true_levels = [1] + true_levels
    expected_error = expected_errors_random_projection(kmax,true_levels)
    # plt.figure()
    # plt.errorbar(Ks, errors, std_errors)
    # plt.plot(Ks, expected_error)
    # expected_error = np.sqrt((Ks - 1)*(kmax-Ks) / (kmax))
    # plt.plot(Ks, expected_error)

    levels, _ = find_all_relevant_minima_from_errors(errors,std_errors,Ks)

    print "\n\nINFERRED LEVELS ONE-SHOT"
    print levels
    for i in (levels-1):
        print partition_vecs[i]

    # relerror = np.zeros(expected_error.size)
    # nonzero = expected_error != 0
    # relerror[nonzero] = (errors[nonzero] - expected_error[nonzero]) / expected_error[nonzero]
    # plt.figure()
    # plt.plot(Ks, relerror)
    # plt.figure()
    # plt.plot(Ks, std_errors)

    print "\n\n START NEW TESTS FULL"
    # WITH INTERMEDIATE AGGREGATION
    pvec_inf = hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                   spectral_oper="Lap", model='SBM',
                                                   reps=10, noise=1e-2, Ks=None)

    print "INFERRED PARTITION2, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]
    for p in pvec_inf:
        print p


def compare_agglomeration_variants_k_given():
    """ Compare various agglomeration techniques"""

    plt.close("all")

    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    n = 3**9
    groups_per_level = 3 
    n_levels = 3
    snr = 3.5
    c_bar = 30
    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    # generate graph and create adjacency
    G = D_actual.generateNetworkExactProb()
    Aorg = D_actual.to_scipy_sparse_matrix(G)
    # get true hierarchy
    true_pvec = D_actual.get_partition_all()
    true_part = [len(np.unique(pv)) for pv in true_pvec][::-1]
    print true_part
    Eagg, Nagg = compute_number_links_between_groups(Aorg, true_pvec[-1])
    Aagg = Eagg / Nagg

    print "Start spec. clust"
    pvec_inf = spectral.hier_spectral_partition(Aorg,
                                                spectral_oper="Lap",
                                                first_pass="Bethe",
                                                model='SBM',
                                                reps=10,
                                                noise=1e-2,
                                                Ks=true_part)
    score_matrix = metrics.calculate_level_comparison_matrix(pvec_inf, true_pvec)
    print score_matrix
    precision, recall = metrics.calculate_precision_recall(score_matrix)
    bottom_lvl = score_matrix[-1, -1]
    print "\n\nRESULTS\n\nbottom level"
    print bottom_lvl
    print len(pvec_inf), len(true_pvec)
    print "precision, recall"
    print precision, recall

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

    print "\n\n START NEW TESTS FULL"
    # WITH INTERMEDIATE AGGREGATION
    pvec_inf = hier_spectral_partition(Aorg, spectral_oper="Lap",
                                                first_pass="Bethe",
                                                model='SBM',
                                                reps=10,
                                                noise=1e-2,
                                                Ks=true_part)

    score_matrix = metrics.calculate_level_comparison_matrix(pvec_inf, true_pvec)
    print score_matrix
    precision, recall = metrics.calculate_precision_recall(score_matrix)
    bottom_lvl = score_matrix[-1, -1]
    print "\n\nRESULTS\n\nbottom level"
    print bottom_lvl
    print len(pvec_inf), len(true_pvec)
    print "precision, recall"
    print precision, recall

    print "INFERRED PARTITION2, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]
    for p in pvec_inf:
        print p

    print "\n\n START NEW TESTS 2"
    # WITH INTERMEDIATE AGGREGATION
    print true_part[0]
    print true_pvec[0]
    pvec_inf = spectral.spectral_partition(Aorg, spectral_oper="Lap", num_groups=true_part[0])

    score_matrix = metrics.overlap_score(true_pvec[0],pvec_inf)
    print score_matrix



def compare_agglomeration_variants_full():
    """ Compare various agglomeration techniques"""

    plt.close("all")

    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    n = 3**9
    groups_per_level = 3 
    n_levels = 3
    snr = 4
    c_bar = 30
    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    # generate graph and create adjacency
    G = D_actual.generateNetworkExactProb()
    Aorg = D_actual.to_scipy_sparse_matrix(G)
    # get true hierarchy
    true_pvec = D_actual.get_partition_all()
    Eagg, Nagg = compute_number_links_between_groups(Aorg, true_pvec[-1])
    Aagg = Eagg / Nagg

    pvec_inf = spectral.hier_spectral_partition(Aorg,
                                                spectral_oper="Lap",
                                                first_pass="Bethe",
                                                model='SBM',
                                                reps=10,
                                                noise=1e-2,
                                                Ks=None)
    score_matrix = metrics.calculate_level_comparison_matrix(pvec_inf, true_pvec)
    print score_matrix
    precision, recall = metrics.calculate_precision_recall(score_matrix)
    bottom_lvl = score_matrix[-1, -1]
    print "\n\nRESULTS\n\nbottom level"
    print bottom_lvl
    print len(pvec_inf), len(true_pvec)
    print "precision, recall"
    print precision, recall

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

    print "\n\n\n\n\n START NEW TESTS FULL"
    # WITH INTERMEDIATE AGGREGATION
    pvec_inf = hier_spectral_partition(Aorg, spectral_oper="Lap",
                                                first_pass="Bethe",
                                                model='SBM',
                                                reps=10,
                                                noise=1e-2,
                                                Ks=None)

    score_matrix = metrics.calculate_level_comparison_matrix(pvec_inf, true_pvec)
    print score_matrix
    precision, recall = metrics.calculate_precision_recall(score_matrix)
    bottom_lvl = score_matrix[-1, -1]
    print "\n\nRESULTS\n\nbottom level"
    print bottom_lvl
    print len(pvec_inf), len(true_pvec)
    print "precision, recall"
    print precision, recall

    print "INFERRED PARTITION2, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]
    for p in pvec_inf:
        print p
