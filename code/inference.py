#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:


"""
import numpy as np
from spectral_operators import BetheHessian, UniformRandomWalk
import cluster
from scipy import linalg


def setup_parameters():
    parameters = {}
    parameters['reps'] = 20
    parameters['noise'] = 2e-2
    parameters['BHnorm'] = False
    parameters['Lnorm'] = True
    return parameters


def infer_hierarchy(A, n_groups=None, parameters=setup_parameters()):
    print('START')
    # get parameters
    reps = parameters['reps']
    Lnorm = parameters['Lnorm']
    BHnorm = parameters['BHnorm']
    noise = parameters['noise']

    # STEP 1: find inital partition
    if n_groups is None:
        K = -1
    else:
        K = n_groups

    initial_partition = cluster_with_BetheHessian(A, num_groups=K,
                                                  regularizer="BHa",
                                                  norm=BHnorm)
    # if n_groups is None:
    n_groups = np.arange(1, initial_partition.k+1)
    # find_levels = True

    hierarchy = cluster.Hierarchy(initial_partition)
    # list_candidates = n_groups

    partition = initial_partition
    Eagg, Nagg = partition.count_links_between_groups(A)
    Aagg = Eagg / Nagg
    selected = np.inf
    while selected > 1:
        print([p.k for p in hierarchy])
        _, part_list = identify_partitions_and_errors(Aagg, n_groups,
                                                      [], 'Fnew',
                                                      norm=Lnorm)
        all_errors = np.zeros((len(n_groups), reps))

        # threshold values close to 0 or 1
        Aagg_ = np.copy(Aagg)
        Aagg_[Aagg < 1e-16] = 1e-16
        Aagg_[Aagg > 1-1e-16] = 1-1e-16
        # repeat with noise
        for kk in range(reps):
            # Anew = cluster.add_noise_to_small_matrix(Aagg, snr=noise)
            Anew = add_noise_to_small_matrix(Aagg_, noise)
            errors, _ = identify_partitions_and_errors(Anew, n_groups,
                                                       part_list, 'Fnew',
                                                       norm=Lnorm)
            all_errors[:, kk] = errors

        mean_errors = np.mean(all_errors, axis=1)
        selected = find_relevant_minima(mean_errors, n_groups)
        # discard partitions that we don't need
        partition = [partition for partition in part_list
                     if partition.k == selected][0]

        if selected > 1:

            hierarchy.add_level(partition)

            k = hierarchy[-1].k
            n_groups = np.arange(1, k+1)

            Eagg, Nagg = hierarchy.count_links_between_groups(Eagg)
            Aagg = Eagg / Nagg

    hierarchy.expand_partitions_to_full_graph()
    return hierarchy


def add_noise_to_small_matrix(M, snr=1e-2, undirected=True):
    var = 1e-5
    normM = linalg.norm(M, 2)
    # initialise random number generator
    rg = np.random.default_rng()
    # calculate N
    N = ((1 - M) * M) / var - 1
    # prevent negative N
    N[N < 2] = 2
    alpha = N * M
    beta = N - alpha
    # Anew = rg.beta(alpha, beta)
    # Anew[np.tril_indices_from(Anew)] = Anew.T[np.tril_indices_from(Anew)]
    noise = np.triu(rg.beta(alpha, beta) - M)
    noise = noise + noise.T - np.diag(np.diag(noise))
    normNoise = linalg.norm(noise, 2)
    Mp = M + snr * normM / normNoise * noise
    return Mp


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


def identify_partitions_and_errors(A, n_groups,
                                   partition_list,
                                   proj_norm='Fnew',
                                   norm=True):
    """
    Collect the partitions and projection errors found for a list n_groups of
    'putative' group numbers
    """
    max_k = np.max(n_groups)

    L = UniformRandomWalk(A)

    # get eigenvectors
    L.find_k_eigenvectors(max_k, which='LM')

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

    k_new = levels[-1]
    # error_reduced = True
    candidates = [1]
    # improvement = [0]
    while k_new > 1:
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
            # print(f'{k}, {total_err[ki]: .3f}, {sigma: .2f}',
            #       f'{1-total_err[ki]/old_diff: .2f}')

        # eliminate levels already included
        total_err[np.array(levels)-1] = np.inf
        # greedy selection: only consider k higher than the last k
        # cum_mean_error[:k_new] = np.inf
        # select level with min cumulative error
        # this favours selection of coarser partitions first.
        idx = np.argmin(total_err)
        # calculate percentage improvement
        improved_by = 1-total_err[idx]/old_diff
        k_new = n_groups[idx] ** (improved_by > -1)
        levels.append(k_new)
        levels.sort()

        # check total error is reduced
        error_reduced = old_diff > total_err[idx]

        if error_reduced:
            # improved_by = 1-total_err[idx]/old_diff
            old_diff = total_err[idx]
            # add levels to candidates
            candidates.append(k_new)
            # improvement.append(improved_by)
        # print('new k', k_new)  # , improved_by, n_groups[np.argmin(cum_mean_error)])
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
