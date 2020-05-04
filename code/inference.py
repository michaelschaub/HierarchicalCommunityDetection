#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:


"""
import numpy as np
from spectral_operators import BetheHessian, Laplacian
import cluster
from scipy.signal import argrelmin


def setup_parameters():
    parameters = {}
    parameters['reps'] = 10
    parameters['noise'] = 2e-2
    parameters['BHnorm'] = False
    parameters['Lnorm'] = True
    parameters['error_threshold'] = 0.2
    return parameters


def infer_hierarchy(A, n_groups=None, parameters=setup_parameters()):

    # get parameters
    reps = parameters['reps']
    noise = parameters['noise']
    Lnorm = parameters['Lnorm']
    BHnorm = parameters['BHnorm']
    error_threshold = parameters['error_threshold']

    # STEP 1: find inital partition
    if n_groups is None:
        K = -1

    initial_partition = cluster_with_BetheHessian(A, num_groups=K,
                                                  regularizer="BHa",
                                                  norm=BHnorm)
    find_levels = False
    if n_groups is None:
        n_groups = np.arange(1, initial_partition.k+1)
        find_levels = True

    hierarchy = cluster.Hierarchy(initial_partition)
    list_candidates = n_groups

    partition = initial_partition
    Eagg, Nagg = partition.count_links_between_groups(A)
    Aagg = Eagg / Nagg
    while len(n_groups) > 0:
        print([p.k for p in hierarchy])
        if find_levels:
            partition_list, all_errors = identify_next_level(Aagg, n_groups,
                                                             proj_norm='Fnew',
                                                             reps=reps,
                                                             noise=noise,
                                                             norm=Lnorm)
            max_errors = np.max(all_errors, axis=1)
            levels = find_relevant_minima(max_errors, n_groups,
                                                        error_threshold)
            selected = np.intersect1d(levels, list_candidates)

            selected = selected[:-1]
            n_groups = n_groups[selected]
            # discard partitions that we don't need
            partition_list = [partition for partition in partition_list
                              if partition.k in selected]

        else:
            raise NotImplementedError('Using specified levels not implemented')

        try:
            partition = partition_list[-1]
            hierarchy.add_level(partition)

            if find_levels:
                # QUESTION: do we use ind_levels_across_agg ?
                # if ind_levels_across_agg:
                k = hierarchy[-1].k
                n_groups = np.arange(1, k+1)
                list_candidates = n_groups

            # if levels are not prespecified, reset candidates
            if k == 1:
                n_groups = []

            Eagg, Nagg = hierarchy.count_links_between_groups(Eagg)
            Aagg = Eagg / Nagg
        # this exception occurs *no candidate* partition (selected == [])
        # and indicates that agglomeration has stopped
        except IndexError:
            print("selected == [] ", selected)

    hierarchy.expand_partitions_to_full_graph()
    return hierarchy


def cluster_with_BetheHessian(A, num_groups=-1, regularizer='BHa',
                              clustermode='KM', norm=False):
    """
    Perform one round of spectral clustering using the Bethe Hessian
    """
    # test that graph contains edges
    if A.sum() == 0:
        partition_vector = np.zeros(A.shape[0], dtype='int')
        return partition_vector

    # construct both the positive and the negative variant of the BH
    BH_pos = BetheHessian(A, regularizer)
    BH_neg = BetheHessian(A, regularizer+'n')

    if num_groups == -1:
        Kpos = BH_pos.find_negative_eigenvectors()
        Kneg = BH_neg.find_negative_eigenvectors()
        combined_evecs = np.hstack([BH_pos.evecs, BH_neg.evecs])

        num_groups = Kpos + Kneg
        print(f'number of groups = {num_groups}')

        if num_groups == 0:
            print("no indication for grouping -- return all in one partition")
            partition_vector = np.zeros(A.shape[0], dtype='int')
            return partition_vector

    else:
        # TODO: note that we combine the eigenvectors of pos/negative BH and
        # do not use information about positive / negative assortativity here
        # find eigenvectors corresponding to the algebraically smallest (most
        # neg.) eigenvalues
        BH_pos.find_k_eigenvectors(num_groups, which='SA')
        BH_neg.find_k_eigenvectors(num_groups, which='SA')

        # combine both sets of eigenvales and eigenvectors and take first k
        combined_evecs = np.hstack([BH_pos.evecs, BH_neg.evecs])
        combined_evals = np.hstack([BH_pos.evals, BH_neg.evals])
        index = np.argsort(combined_evals)
        combined_evecs = combined_evecs[:, index[:num_groups]]

    part = cluster.find_partition(combined_evecs, num_groups,
                                  method=clustermode, normalization=norm)

    return part


def identify_next_level(A, n_groups, proj_norm='Fnew',
                        reps=20, noise=2e-2, norm=True):
    """
    Identify agglomeration levels by checking the projection errors and
    comparing the to a perturbed verstion of the same network
    Inputs:
        A -- (agglomerated) adjacency matrix of network
        n_groups -- list of the partition sizes at each level
              (e.g. [3 9, 27] for a hierarchical split into 3 x 3 x 3 groups.
        proj_norm -- norm used to assess projection error
        reps -- number of repetitions for bootstrap
        noise -- noise parameter for bootstrap

    Outputs:
        partition_list -- found putative hier partitions
        all_errors -- projection errors from perturbed A.

    """
    # first identify partitions and their projection error
    mean_errors, partition_list = identify_partitions_and_errors(A, n_groups,
                                                                 [], proj_norm,
                                                                 norm=norm)
    all_errors = np.zeros((len(n_groups), reps+1))

    # repeat with noise
    for kk in range(reps):
        Anew = cluster.add_noise_to_small_matrix(A, snr=noise)
        errors, _ = identify_partitions_and_errors(Anew, n_groups,
                                                   partition_list, proj_norm,
                                                   norm=norm)
        all_errors[:, kk] = errors

    # include actual projection error in all_errors
    all_errors[:, -1] = mean_errors
    return partition_list, all_errors


def identify_partitions_and_errors(A, n_groups,
                                   partition_list,
                                   proj_norm='Fnew',
                                   norm=True):
    """
    Collect the partitions and projection errors found for a list n_groups of
    'putative' group numbers
    """
    max_k = np.max(n_groups)

    L = Laplacian(A)

    # get eigenvectors
    L.find_k_eigenvectors(max_k, which='SM')

    # initialise errors
    error = np.zeros(len(n_groups))

    if partition_list == []:
        # find partitions and their error for each k
        for k in n_groups:
            partition = cluster.find_partition(L.evecs, k, normalization=norm)
            partition_list.append(partition)

    for ki, partition in enumerate(partition_list):
        # calculate and store error
        error[ki] = partition.calculate_proj_error(L.evecs, proj_norm)

    return error, partition_list


def find_relevant_minima(errors, n_groups, threshold):
    """Given a set of error and standard deviations, find best minima"""

    levels = [1, errors.size]

    total_err = np.empty(len(errors))
    k_new = 2
    while k_new > 1:
        for ki, k in enumerate(n_groups):

            levels_i = levels + [k]
            levels_i.sort()
            expected_errors = expected_errors_random_projection(levels_i)
            total_err[ki] = np.sum((errors - expected_errors) ** 2)

        k_new = n_groups[np.argmin(total_err)]
        print('Found level', k_new)
        levels.append(k_new)

    # expected_error = expected_errors_random_projection(levels)
    # next_level, below_thresh = find_smallest_relevant_minima(errors,
    #                                                          expected_error,
    #                                                          threshold)
    # while next_level != -1:
    #     levels = levels + [next_level]
    #     levels.sort()
    #     expected_error = expected_errors_random_projection(levels)
    #     next_level, _ = find_smallest_relevant_minima(errors, expected_error,
    #                                                   threshold)

    # QUESTION: why might we have zero here?
    # remove again the level 1 entry
    if levels[0] == 1:
        levels = levels[1:]

    return np.array(levels)


def expected_errors_random_projection(levels):
    """
    Compute vector of expected errors for random projection
        dim_n -- ambient space
        levels of hierarchy [1, ..., dim_n]
    """
    start_end_pairs = zip(levels[:-1], levels[1:])

    expected_error = []
    for i, j in start_end_pairs:
        Ks = np.arange(i, j)
        errors = (Ks - i) * (j - Ks) / (j-i)
        expected_error = np.hstack([expected_error, errors])
    expected_error = np.hstack([expected_error, 0])
    return expected_error


def find_smallest_relevant_minima(errors, expected_error, threshold=0.2):
    relerror = np.zeros(expected_error.size)
    ratio_error = np.zeros(expected_error.size)

    # find points below relative error
    nonzero = expected_error != 0
    relerror[nonzero] = (errors[nonzero]) / expected_error[nonzero]
    below_thresh = np.nonzero(relerror < threshold)[0]

    # find relative minima
    ratio_error[nonzero] = (errors[nonzero]) / expected_error[nonzero]
    local_min = argrelmin(ratio_error)[0]

    # levels == below thres && local min
    levels = np.intersect1d(local_min, below_thresh).astype(int)

    # remove already found local minima from list
    Ks = np.arange(errors.size)
    levels = np.intersect1d(levels, Ks[nonzero]).astype(int)

    Ks = np.arange(1, errors.size+1)
    best_level = -1
    if levels.size > 0:
        best_level = levels[np.argmin(relerror[levels])]+1

    return best_level, below_thresh
