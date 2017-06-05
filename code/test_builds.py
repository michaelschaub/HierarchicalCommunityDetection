from __future__ import division
import numpy as np
import GHRGbuild


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
    av_deg=15
    groups_per_level = 2
    n_levels = 2
    Nr=n*(n-1)
    D=GHRGbuild.create2paramGHRG(n,snr,av_deg,n_levels,groups_per_level)
    
    # top level
    nl=n/groups_per_level
    v=0
    print "root node"
    assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
    assert np.tril(D.node[v]['Er']).sum() == 0
    assert np.tril(D.node[v]['Nr']).sum() == 0
    assert D.node[v]['Nr'][0,1] == nl*nl
    
    # first level
    nl=nl/groups_per_level
    for v in [1,2]:
        print "node", v
        assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
        assert np.tril(D.node[v]['Er']).sum() == 0
        assert np.tril(D.node[v]['Nr']).sum() == 0
        assert D.node[v]['Nr'][0,1] == nl*nl
    
    # second level (leaves)
    for v in xrange(3,7):
        print "node", v
        assert np.sum(D.node[v]['Er']>D.node[v]['Nr']) == 0
        print  D.node[v]['Nr'][0,0], nl*(nl-1)/2
        assert D.node[v]['Nr'][0,0] == nl*(nl-1)/2