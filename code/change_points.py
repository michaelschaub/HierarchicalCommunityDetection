from __future__ import division
import numpy as np
import spectral_algorithms as spectral
import inference
import networkx as nx
import model_selection

def detectChanges_flat(Gs,w,mode='Lap'):
    #create sparse matrices
    As = [nx.to_scipy_sparse_matrix(G) for G in Gs]
    
    #length of sequence
    seq_n = len(Gs)
    
    #sliding window
    for start_idx in xrange(seq_n-w+1):
        end_idx = start_idx + w + 1
        
        #infer model for whole window
        #NOTE: for now we assume number of groups is two and it is known
        Aw=sum(As[start_idx:end_idx])
        
        D_inf = inference.split_network_by_recursive_spectral_partition(Aw, mode=mode, num_groups=2, max_depth=0)
        
        #correct Nr for multiple networks
        D_inf.node[0]['Nr']*=w
        
        test_posteriorBayesFactor(D_inf,w)
        
        
    return D_inf
    #~ partition = spectral.spectral_partition(Aw, mode=mode, num_groups=k)
    
    #~ E_rs, N_rs = compute_number_links_between_groups(A,partition)
    


def test_posteriorBayesFactor(D,w):
    partition = D.get_highest_partition()
    
    Gs=[D.generateNetwork() for i in xrange(w)]
    #~ Gs=[D.generateNetworkExactProb() for i in xrange(w)]