from __future__ import division
import numpy as np
import GHRGbuild
import GHRGmodel


def test_create_degree_from_snr_av_deg():
    SNR=0
    av_degree = 15
    num_cluster = 2
    a,b = GHRGbuild.calculateDegreesFromAvDegAndSNR(SNR,av_degree,num_cluster)
    assert a==15
    assert b==15


def test_ErNr_in_create2paramGHRG():
    n=1000
    snr=2
    av_degree=15
    groups_per_level = 2
    n_levels = 2

    a,b = GHRGbuild.calculateDegreesFromAvDegAndSNR(snr,av_degree,groups_per_level)

    D=GHRGbuild.create2paramGHRG(n,snr,av_degree,n_levels,groups_per_level)

    # top level
    nl=n/groups_per_level
    v=0
    print "root node"
    assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
    assert np.tril(D.node[v]['Er']).sum() == 0
    assert np.tril(D.node[v]['Nr']).sum() == 0
    assert D.node[v]['Nr'][0,1] == nl*nl
    assert D.node[v]['Er'][0,1] == b/n*nl*nl

    # first level
    n=nl
    nl=nl/groups_per_level
    for v in [1,2]:
        print "node", v
        assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
        assert np.tril(D.node[v]['Er']).sum() == 0
        assert np.tril(D.node[v]['Nr']).sum() == 0
        assert D.node[v]['Nr'][0,1] == nl*nl
        assert D.node[v]['Er'][0,1] == b/n*nl*nl

    # second level (leaves)
    for v in xrange(3,7):
        print "node", v
        assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
        print  D.node[v]['Nr'][0,0], nl*(nl-1)/2
        assert D.node[v]['Nr'][0,0] == nl*(nl-1)/2
        assert D.node[v]['Er'][0,0] == a/n*nl*(nl-1)/2


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
    partition_true = D_gen.get_lowest_partition()

    D = GHRGmodel.GHRG()
    D.infer_spectral_partition_flat(A)

    partition_inf = D.get_lowest_partition()
    assert sum(partition_inf == partition_true)/n >= 0.995

