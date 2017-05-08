from __future__ import division
import numpy as np
import spectral_algorithms as spectral
import inference
import networkx as nx
import model_selection
from itertools import izip

def detectChanges_flat(Gs,w,mode='Bethe'):
    #create sparse matrices
    As = [nx.to_scipy_sparse_matrix(G) for G in Gs]
    
    #length of sequence
    seq_n = len(Gs)
    
    #sliding window
    for start_idx in xrange(seq_n-w+1):
        end_idx = start_idx + w
        
        #infer model for whole window
        #NOTE: for now we assume number of groups is two and it is known
        Aw=sum(As[start_idx:end_idx])
        
        #~ D_inf = inference.split_network_by_recursive_spectral_partition(Aw, mode=mode, num_groups=2, max_depth=0)
        D_inf = inference.split_network_spectral_partition(Aw,mode=mode,num_groups=2)
        
        #correct Nr for multiple networks
        D_inf.node[0]['Er']/=w
        
        test_posteriorBayesFactor(As[start_idx:end_idx],D_inf,w)
        
        
    return D_inf
    #~ partition = spectral.spectral_partition(Aw, mode=mode, num_groups=k)
    
    #~ E_rs, N_rs = compute_number_links_between_groups(A,partition)
    


def test_posteriorBayesFactor(As,D,w,n_samples=1000):
    partition = D.get_highest_partition()   #NEED TO UPDATE: this only uses highest partition (while we assume number of groups is two and it is known)
    test_statistic = -np.inf
    best_change_point=None
    
    obs_edge_counts = zip(*[spectral.compute_number_links_between_groups(A,partition) for A in As]) #NEED to adjust diagonal blocks? I think Ers are correct...
    # here we use Er as edge counts and Nr for non-edge counts
    obs_Ers=np.array(obs_edge_counts[0])
    obs_Nrs=np.array(obs_edge_counts[1])-obs_Ers
    
    for cp in xrange(1,w):
        #~ before_ec= zip(*obs_edge_counts[:cp])
        #~ after_ec= zip(*obs_edge_counts[cp:])
        a=np.array([1.+sum(obs_Ers[:cp])])
        b=np.array([1.+sum(obs_Nrs[:cp])])
        ts = sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers[:cp]),b=1.+sum(obs_Nrs[:cp])).sum() for (Er,Nr) in izip(obs_Ers[:cp],obs_Nrs[:cp]))
        print a.shape, b.shape, obs_Ers[:cp].shape,obs_Nrs[:cp].shape
        print a,b,ts, model_selection.betabinlik(obs_Ers[:cp],obs_Nrs[:cp],a,b) #,a=1.+sum(obs_Ers[:cp]),b=1.+sum(obs_Nrs[:cp])).sum()
        ts += sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers[cp:]),b=1.+sum(obs_Nrs[cp:])).sum() for (Er,Nr) in izip(obs_Ers[cp:],obs_Nrs[cp:]))
        #~ print adad
        #~ print cp,ts
        if ts>test_statistic:
            best_change_point=cp
            test_statistic=ts
    
    #~ print test_statistic,sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers),b=1.+sum(obs_Nrs)).sum() for (Er,Nr) in izip(obs_Ers,obs_Nrs))
    test_statistic -= sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers),b=1.+sum(obs_Nrs)).sum() for (Er,Nr) in izip(obs_Ers,obs_Nrs))
    print 'most likely change',best_change_point,test_statistic
    
    
    null_dist=np.empty(n_samples)
    
    #generate null distribution
    #TODO - loop over w (instead of n_samples)
    for i in xrange(n_samples):
        
        #~ edge_counts=zip(*[spectral.compute_number_links_between_groups(nx.to_scipy_sparse_matrix(D.generateNetworkExactProb()),partition) for j in xrange(w)])
        edge_counts=zip(*[spectral.compute_number_links_between_groups(nx.to_scipy_sparse_matrix(D.generateNetworkBeta()),partition) for j in xrange(w)])
        
        Ers=np.array(edge_counts[0])
        Nrs=np.array(edge_counts[1])-Ers
        
        ts = sum(model_selection.betabinlik(Er,Nr,a=1.+sum(Ers[:best_change_point]),b=1.+sum(Nrs[:best_change_point])).sum() for (Er,Nr) in izip(Ers[:best_change_point],Nrs[:best_change_point]))
        ts += sum(model_selection.betabinlik(Er,Nr,a=1.+sum(Ers[best_change_point:]),b=1.+sum(Nrs[best_change_point:])).sum() for (Er,Nr) in izip(Ers[best_change_point:],Nrs[best_change_point:]))
        
        null_dist[i] = ts - sum(model_selection.betabinlik(Er,Nr,a=1.+sum(Ers),b=1.+sum(Nrs)).sum() for (Er,Nr) in izip(Ers,Nrs))
        
    #~ print null_dist
    p_value = 1-(test_statistic>null_dist).sum()/n_samples
    
    print 'p value:',p_value
    
    return p_value