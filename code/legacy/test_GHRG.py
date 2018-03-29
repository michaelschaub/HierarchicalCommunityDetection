from __future__ import division
import numpy as np
import GHRGbuild
import GHRGmodel


def test_GHRG_inference_flat():
    # mean degree and number of nodes etc.
    n=1600
    n_groups = 8
    n_levels = 1
    K=n_groups**n_levels
    av_degree = 20


    SNR = 10

    # create GHRG object with specified parameters and create a sample network from it
    D_gen=GHRGbuild.create2paramGHRG(n,SNR,av_degree,n_levels,n_groups)
    G= D_gen.generateNetworkExactProb()
    A= D_gen.to_scipy_sparse_matrix(G)
    partition_true, _ = D_gen.get_partition_at_level(-1)

    D = GHRGmodel.GHRG()
    D.infer_spectral_partition_flat(A)
    n_levels_inferred = D.get_number_of_levels()
    assert (n_levels_inferred == n_levels)
    print D.node[0]

    partition_inf, dendro_leaf_nodes = D.get_partition_at_level(-1)
    assert dendro_leaf_nodes == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sum(partition_inf == partition_true)/n >= 0.995

def test_GHRG_inference_hier():
    # mean degree and number of nodes etc.
    n=1600
    n_levels = 3
    groups_per_level = 2
    K=groups_per_level**n_levels
    av_degree = 30

    snr = 10

    D_gen=GHRGbuild.create2paramGHRG(n,snr,av_degree,n_levels,groups_per_level)
    G= D_gen.generateNetworkExactProb()
    A= D_gen.to_scipy_sparse_matrix(G)

    D = GHRGmodel.GHRG()
    D.infer_spectral_partition_hier(A)
    n_levels_inferred = D.get_number_of_levels()
    assert (n_levels_inferred == n_levels)

    for i in range(n_levels+1):
        partition_true, d_nodes = D_gen.get_partition_at_level(i)
        partition_inf, d_inf_nodes = D.get_partition_at_level(i)
        assert sum(partition_inf == partition_true)/n >= 0.995
        assert d_nodes == d_inf_nodes
