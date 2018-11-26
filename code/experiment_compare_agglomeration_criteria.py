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
from helperfunctions import *

def find_subspace_angle_between_ev_bases(U,V):
    Q1 = U.T.dot(V)
    sigma = scipy.linalg.svd(Q1,compute_uv=False)
    angle = -np.ones(sigma.size)
    cos_index = sigma**2 <=0.5
    if np.all(cos_index):
        angle = np.arccos(sigma)
    else:
        Q2 = V - U.dot(Q1)
        sigma2 = scipy.linalg.svd(Q2,compute_uv=False)
        angle[cos_index] = np.arccos(sigma[cos_index])
        sin_index = np.bitwise_not(cos_index)
        angle[sin_index] = np.arcsin(sigma2[sin_index])

    return angle, sigma


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
    elif norm == '2':
        proj2 = project_orthogonal_to(V, H)
        norm1 = scipy.linalg.norm(proj1, 2)
        norm2 = scipy.linalg.norm(proj2, 2)
        error = .5 * (norm1 + norm2)

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
    meanerror = np.mean(error)
    stderror = np.std(error)

    return error


def hier_spectral_partition_agglomerate(A, partition, spectral_oper="Lap", model='SBM', reps=10, noise=1e-3, Ks=None, no_Ks_forward=True):
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

    # k == index of highest groups / number of groups = k+1
    k = np.max(partition)
    print "HIER SPECTRAL PARTITION -- agglomerative\n"
    print "Initial partition into", k + 1, "groups \n"

    # Ks stores the candidate levels in inverse order
    # Note: set to min 1 group, as no agglomeration required
    # when only 2 groups are detected.
    if Ks is None:
        Ks = np.arange(k, 1, -1)
        find_levels = True
    else:
        find_levels = False

    # levels is a list of 'k' values of each level in the inferred hierarchy
    # pvec stores all hier. refined partitions
    levels = [k + 1]
    pvec = []
    pvec.append(partition)
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
            kmax = Ks[0]
            expected_error = np.sqrt(Ks - Ks**2 / kmax)
            errors[-1] = 1
            errors[0] = 1
            relerror = (errors - expected_error) / errors
            threshold = -2
            local_min = argrelmin(relerror)[0]
            below_thresh = np.nonzero(relerror < threshold)[0]
            print "HERE"
            print Ks, Ks[local_min], Ks[below_thresh]
            selected = np.intersect1d(local_min, below_thresh).astype(int)
            Ks = Ks[selected]
            print "Minima at", Ks
            hier_partition_vecs = [hier_partition_vecs[si] for si in selected]
        else:
            k = Ks[0] - 1
            Ks, hier_partition_vecs = identify_partitions_at_level(
                Aagg, Ks, model=model, reg=False, norm='F')

        try:
            pvec.append(hier_partition_vecs[0])
            partition = expand_partitions_to_full_graph(pvec)[-1]

            # TODO: it might be useful to pass down candidates
            # from previous agglomeration rounds here instead of starting from scratch!
            if find_levels:
                if no_Ks_forward:
                    k = Ks[0] - 1
                    Ks = np.arange(k, 1, -1)
                else:
                    # TODO implement this!
                    pass
            levels.append(k + 1)
            # if levels are not prespecified, reset candidates
            print 'partition into', k + 1, ' groups'
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

    return expand_partitions_to_full_graph(pvec)[::-1]


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

        for rep in xrange(reps):
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
    tau = 0

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


def create_agglomerated_graphGHRG():

    groups_per_level = 3
    n_levels = 3
    n = 3**7
    c_bar = 50
    snr = 10
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

    # Create an aggregated graph as we would have in the test loop
    # and check current agglomeration practise
    Aagg, Aorg, true_pvec = create_agglomerated_graphGHRG()
    pvec_inf = spectral.hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                            spectral_oper="Lap",
                                                            model='SBM',
                                                            reps=10,
                                                            noise=1e-3,
                                                            Ks=None)
    print "INFERRED PARTITION, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]

    plt.close('all')
    plt.figure()
    plt.imshow(Aagg-np.diag(np.diag(Aagg)))


    # ############
    # Variant 2:
    # get raw projeciton errors and compare to analytical value in 'relative' manner
    kmax, _ = Aagg.shape
    Ks = np.arange(1, kmax+1)
    errors, std_errors, partition_vecs = identify_next_level(Aagg, Ks)

    expected_error = np.sqrt(Ks - Ks**2 / kmax)
    candidates = expected_error - (errors + 2*std_errors) > 0
    print errors
    plt.figure()
    plt.errorbar(Ks, errors, std_errors)
    plt.plot(Ks, expected_error)

    # set first and last error to 1 to avoid division by zero
    # these are not local minima in any case
    errors[-1] = 1
    errors[0] = 1
    relerror = (errors - expected_error) / errors
    #TODO: how to choose the threshold?
    threshold = -2
    local_min = argrelmin(relerror)[0] + 1
    below_thresh = np.nonzero(relerror < threshold)[0] + 1
    levels = np.intersect1d(local_min, below_thresh).astype(int)

    print levels
    for i in (levels-1):
        print partition_vecs[i]

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

################
# UNTESTED STUFF
################
def test_agglomeration_variants():
    """ Compare various agglomeration techniques"""

    # Create an aggregated graph as we would have in the test loop
    Aagg, Aorg, true_pvec = create_agglomerated_graphGHRG()
    pvec_inf = spectral.hier_spectral_partition_agglomerate(Aorg, true_pvec[-1],
                                                            spectral_oper="Lap",
                                                            model='SBM',
                                                            reps=10,
                                                            noise=1e-3,
                                                            Ks=None)
    print "INFERRED PARTITION, TRUE PARTITION"
    print[len(np.unique(pv)) for pv in pvec_inf]
    print[len(np.unique(pv)) for pv in true_pvec]

    kmax, _ = Aagg.shape
    Ks = np.arange(kmax, 1, -1)
    errors, std_errors, partition_vecs = identify_next_level(Aagg, Ks)
    errors2 = np.zeros(kmax)
    std_errors2 = np.zeros(kmax)
    for i in Ks:
        error = test_random_projection_with_kmeans2(kmax, i)
        errors2[i-1] = np.mean(error)
        std_errors2[i-1] = np.std(error)

    expected_error = np.sqrt(Ks - Ks**2 / kmax)
    candidates = expected_error - (errors + 2*std_errors) > 0
    minima_errors = np.zeros_like(errors)
    minima_errors[~candidates] = 9999
    minima_errors[candidates] = errors[candidates]
    print minima_errors
    candidates = np.nonzero(candidates[::-1] > 0)[0]
    print "Candidates for hierarchy:"
    print candidates
    minima_errors = np.array([0.0001] + [e for e in minima_errors[::-1]])
    print minima_errors
    minima = argrelmin(minima_errors)[0] + 1
    print minima
    relminima_errors = (minima_errors - errors2) / minima_errors
    plt.figure()
    plt.plot(relminima_errors)
    minima2 = argrelmin(relminima_errors)[0] + 1
    print minima2

    plt.figure()
    plt.errorbar(Ks, errors, 2*std_errors)
    plt.errorbar(np.arange(1, kmax + 1), errors2, std_errors2)
    plt.plot(Ks, np.sqrt(Ks - Ks**2 / kmax))

    plt.figure()
    meas_errors = np.array([0] + [e for e in errors[::-1]])
    meas_errors[meas_errors == 0] = 1
    errors2[errors2 == 0] = 1
    std_errors2[0] = 0
    plt.plot(np.arange(1, kmax + 1), (meas_errors - errors2) / meas_errors)

    plt.figure()
    meas_errors = np.array([0] + [e for e in errors[::-1]])
    meas_errors[meas_errors == 0] = 1
    errors2[errors2 == 0] = 1
    std_errors2[0] = 0
    meas_errors2 = np.array([0] + [e for e in std_errors[::-1]])
    plt.plot(np.arange(1, kmax + 1), meas_errors2)

def test_agglomeration_ideas_noise_pert(groups_per_level=3):
    # n=2**13
    n=3**9
    snr=10
    c_bar=50
    n_levels=3

    max_k = groups_per_level**n_levels
    norm = 'F'
    mode = 'SBM'
    thres = 1/3

    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1) # true partition lowest level
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)

    # do a first round of clustering with the Bethe Hessian
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=groups_per_level**n_levels,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='GMM')
    # p0 = p0.astype(int)
    p0=ptrue.astype(int)
    # k0 = p0.max() + 1
    # p0, _ = spectral.regularized_laplacian_spectral_clustering(A,num_groups=k0,clustermode='qr')
    p0 = spectral.relabel_partition_vec(p0)
    p0 = p0.astype(int)
    plt.figure(1)
    plt.plot(p0,'x')

    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / Nagg
    aggregate = True

    while aggregate:
        max_k = np.max(p0)+1
        error = np.zeros(max_k)
        angle_min = np.zeros(max_k)
        angle_max = np.zeros(max_k)
        likelihood = np.zeros(max_k)
        print "\n\nAggregation round started"
        print("maximal k ", max_k)
        reg= False
        # normalized Laplacian is D^-1/2 A D^-1/2
        L = spectral.construct_graph_Laplacian(Aagg)
        tau = 0
        ev, evecs = scipy.linalg.eigh(L)

        index = np.argsort(np.abs(ev))
        evecs = evecs[:,index]
        sigma = np.abs(ev[index])

        candidates_for_hier = np.zeros(max_k)
        for k in range(max_k):

            partition_vec, Hnorm = spectral.find_partition(evecs, k+1, mode, 0)
            H = spectral.create_partition_matrix_from_vector(partition_vec)
            error[k] = calculate_proj_error(evecs, Hnorm, norm)
            angles, _ = find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k+1])
            angle_min[k] = np.min(angles)
            angle_max[k] = np.max(angles)
            print("K, error / exp rand error, likelihood")
            print(k+1, error[k], likelihood[k])

            if error[k] - thres < 0:
                candidates_for_hier[k] = 1

        plt.figure(4)
        plt.plot(1+np.arange(max_k),angle_max)
        plt.plot(1+np.arange(max_k),angle_min)

        candidate_list = np.nonzero(candidates_for_hier)[0]+1
        print "\ninitial candidate_list: "
        print candidate_list
        print "\n\ncreating perturbed samples"
        num_pert = 10
        error_rand = np.ones((num_pert,max_k))
        angle_max_rand = np.zeros((num_pert,max_k))
        angle_min_rand = np.zeros((num_pert,max_k))
        likelihood_rand = np.zeros((num_pert,max_k))
        for pp in range(num_pert):
            Anew = add_noise_to_small_matrix(Aagg)
            L = spectral.construct_graph_Laplacian(Anew)

            ev, evecs = scipy.linalg.eigh(L)
            index = np.argsort(np.abs(ev))
            evecs = evecs[:,index]
            sigma = np.abs(ev[index])

            for k in candidate_list:

                partition_vec, Hnorm = spectral.find_partition(evecs, k, mode, 0)
                error_rand[pp,k-1] = calculate_proj_error(evecs, Hnorm, norm)
                angle_min_rand[pp,k-1] = np.min(find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k])[0])
                angle_max_rand[pp,k-1] = np.max(find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k])[0])
                likelihood_rand[pp,k-1] = compute_likelihood_SBM(partition_vec[p0],A)
                print("K, error / exp rand error, likelihood")
                print(k, error_rand[pp,k-1], likelihood_rand[pp,k-1])

        error_av = np.mean(error_rand,0)
        error_std = np.std(error_rand,0)
        plt.figure(2)
        plt.plot(1+np.arange(max_k),error_av)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')
        plt.figure(3)
        plt.plot(1+np.arange(max_k),error_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')

        angle_min_rand_av = np.mean(angle_min_rand,0)
        angle_max_rand_av = np.mean(angle_max_rand,0)
        angle_min_rand_std = np.std(angle_min_rand,0)
        angle_max_rand_std = np.std(angle_max_rand,0)
        plt.figure(5)
        plt.errorbar(1+np.arange(max_k),angle_min_rand_av,yerr=angle_min_rand_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')
        plt.figure(6)
        plt.errorbar(1+np.arange(max_k),angle_max_rand_av,yerr=angle_max_rand_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')


        relative_minima = argrelmin(error_av)[0] + 1
        print "Relative minima"
        print relative_minima
        filtered_candidates_local_minima = np.intersect1d(relative_minima, candidate_list)
        filter_start = 0*np.nonzero(np.diff(candidates_for_hier)==-1)[0]+1
        print "Filter start"
        print filter_start
        filtered_candidates = np.union1d(filtered_candidates_local_minima,filter_start)
        # filtered_candidates = filtered_candidates_local_minima
        filtered_candidates = np.setdiff1d(filtered_candidates,np.ones(1))
        print "Candidate levels for merging"
        print filtered_candidates

        found_partition = False

        for k in filtered_candidates[::-1]:
            print k
            std = scipy.std(error_rand[:,k-1])
            if std  < 0.01 :
                print "std: ", std
                print "\nAgglomeration Test passed, at level\n", k
                found_partition = True
                partition_vec = spectral.find_partition(evecs,k,mode,0)
                p0 = partition_vec[p0]
                p0 = spectral.relabel_partition_vec(p0)
                plt.figure(1)
                plt.plot(p0,'-')
                break


        if found_partition:
            Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
            Aagg = Eagg / np.sqrt(Nagg)
            aggregate = True
        else:
            aggregate = False

    return D_actual
