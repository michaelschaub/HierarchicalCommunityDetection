#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:


"""
import numpy as np
from spectral_operators import BetheHessian, Laplacian
import cluster
from scipy import linalg


def setup_parameters():
    parameters = {}
    parameters['reps'] = 10
    parameters['noise'] = 2e-2
    parameters['BHnorm'] = False
    parameters['Lnorm'] = True
    return parameters


def infer_hierarchy(A, n_groups=None, parameters=setup_parameters()):
    print('START')
    # get parameters
    reps = parameters['reps']
    noise = parameters['noise']
    Lnorm = parameters['Lnorm']
    BHnorm = parameters['BHnorm']

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
    # list_candidates = n_groups

    partition = initial_partition
    Eagg, Nagg = partition.count_links_between_groups(A)
    Aagg = Eagg / Nagg
    selected = np.inf
    while selected > 1:
        print([p.k for p in hierarchy])
        if find_levels:
            partition_list, all_errors = identify_next_level(Aagg, n_groups,
                                                             proj_norm='Fnew',
                                                             reps=reps,
                                                             noise=noise,
                                                             norm=Lnorm)
            max_errors = np.mean(all_errors[:, :-1], axis=1)
            selected = find_relevant_minima(max_errors, n_groups)
            # discard partitions that we don't need
            partition = [partition for partition in partition_list
                         if partition.k == selected][0]

        else:
            raise NotImplementedError('Using specified levels not implemented')

        if selected > 1:

            hierarchy.add_level(partition)

            if find_levels:
                # QUESTION: do we use ind_levels_across_agg ?
                # if ind_levels_across_agg:
                k = hierarchy[-1].k
                n_groups = np.arange(1, k+1)
                # list_candidates = n_groups

            Eagg, Nagg = hierarchy.count_links_between_groups(Eagg)
            Aagg = Eagg / Nagg

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


def find_relevant_minima(errors, n_groups):
    """Given a set of mean perturbed errors find the next candidate level"""

    # initialise levels
    levels = [1, errors.size]
    # calculate difference between error and expected errors
    expected_errors = expected_errors_random_projection(levels)
    old_diff, sigma = match_curves(errors, expected_errors)
    # old_diff_this_level = old_diff
    # old_diff = linalg.norm(errors - expected_errors, 2)

    k_new = 0
    error_reduced = True
    candidates = [1]
    # improvement = [0]
    while error_reduced:
        total_err = np.empty(len(errors))
        # cum_mean_error = np.empty(len(errors))
        # print('old_diff', old_diff)
        for ki, k in enumerate(n_groups):
            # consider partition into k groups as a candidate level
            levels_i = levels + [k]
            levels_i.sort()
            # calculate expected errors conditioned on k
            expected_errors = expected_errors_random_projection(levels_i)
            total_err[ki], sigma = match_curves(errors, expected_errors)
            # diff = errors - sigma*expected_errors
            # calculate cumulative and total error difference
            # cum_mean_error[ki] = linalg.norm(diff[:k], 2)/k
            # total_err[ki] = linalg.norm(diff, 2)
            # print(k, cum_mean_error[ki], total_err[ki], sigma, 1-total_err[ki]/old_diff)

        # eliminate levels already included
        # cum_mean_error[np.array(levels)-1] = np.inf
        # greedy selection: only consider k higher than the last k
        # cum_mean_error[:k_new] = np.inf
        # select level with min cumulative error
        # this favours selection of coarser partitions first.
        idx = np.argmin(total_err)
        k_new = n_groups[idx]
        # calculate percentage improvement
        # improved_by = 1-total_err[idx]/old_diff_this_level

        # check total error is reduced
        error_reduced = old_diff > total_err[idx]

        if error_reduced:
            # improved_by = 1-total_err[idx]/old_diff
            old_diff = total_err[idx]
            # add levels to candidates
            levels.append(k_new)
            levels.sort()
            candidates.append(k_new)
            # improvement.append(improved_by)
        # print('new k', k_new, improved_by, n_groups[np.argmin(cum_mean_error)])
    # return the candidate level that offers best improvement
    # return candidates[np.argmax(improvement)]
    return np.max(candidates)


def match_curves(curve1, curve2, delta=1, log=True):

    log_curve1 = np.log(curve1 + delta)
    # if log:
    error, sigma = min([(linalg.norm(log_curve1
                        - np.log(ii*curve2 + delta), 2), ii)
                        for ii in np.linspace(0, 1, 101)])
    # else:
    #     print('no log')
    #     error, sigma = min([(linalg.norm(curve1 - ii*curve2, 2), ii)
    #                         for ii in np.linspace(0, 1, 101)])
    return error, sigma


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
