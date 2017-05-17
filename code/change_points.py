from __future__ import division
import numpy as np
import spectral_algorithms as spectral
import inference
import networkx as nx
import model_selection
from itertools import izip


"""
Things to try out:
----------------------

test statistics:
-Post. Bayes Factor (Peel&Clauset2015)
-Looxv

model inference:
-fixed dendro (Peel&Clauset2015)
-re-infer dendro (before, after, no-change) -- feasible now that 

"""

def detectChanges_flat(Gs,w):
    """
    Main function for detecting changes based on a sliding window along a sequence of networks Gs.
    """
    from matplotlib import pyplot as plt
    plt.ion()
    plt.close('all')
    
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

        D_inferred = inference.split_network_spectral_partition(Aw)
        
        #correct Nr for multiple networks  
        #~ D_inferred.node[0]['Er']/=w     # NEED TO UPDATE: this only updates the top node!!!
        
        p,ts,nd = test_posteriorBayesFactor(As[start_idx:end_idx],D_inferred,w)
        
        plt.figure()
        plt.hist(nd)
        plt.axvline(ts)
        
        
    return D_inferred
    #~ partition = spectral.spectral_partition(Aw, mode=mode, num_groups=k)
    
    #~ E_rs, N_rs = compute_number_links_between_groups(A,partition)
    #~ G= D_gen.generateNetworkExactProb()
    #~ A= D_gen.to_scipy_sparse_matrix(G)

    #~ D_inferred = inference.split_network_spectral_partition(A)


def test_posteriorBayesFactor(As,D,w,n_samples=1000,directed=False):
    partition = D.get_lowest_partition()  
    print "number of communities", len(np.unique(partition))
    
    test_statistic = -np.inf
    best_change_point=None
    
    obs_edge_counts = zip(*[spectral.compute_number_links_between_groups(A,partition,directed=directed) for A in As]) #NEED to create function within GHRG
    
    obs_Ers=np.array(obs_edge_counts[0])
    obs_1_Ers=np.array(obs_edge_counts[1])-obs_Ers
    
    
    for cp in xrange(1,w):
        
        #calculate test statistic
        a0=(1.+np.sum(obs_Ers[:cp],0))[np.newaxis,:,:]
        a1=(1.+np.sum(obs_Ers[cp:],0))[np.newaxis,:,:]
        b0=(1.+np.sum(obs_1_Ers[:cp],0))[np.newaxis,:,:]
        b1=(1.+np.sum(obs_1_Ers[cp:],0))[np.newaxis,:,:]
        #~ ts = sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers[:cp]),b=1.+sum(obs_Nrs[:cp])).sum() for (Er,Nr) in izip(obs_Ers[:cp],obs_Nrs[:cp]))
        #~ ts += sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers[cp:]),b=1.+sum(obs_Nrs[cp:])).sum() for (Er,Nr) in izip(obs_Ers[cp:],obs_Nrs[cp:]))
        ts = model_selection.betabinlik(obs_Ers[:cp],obs_1_Ers[:cp],a0,b0).sum()
        ts += model_selection.betabinlik(obs_Ers[cp:],obs_1_Ers[cp:],a1,b1).sum()
        
        if ts>test_statistic:
            best_change_point=cp
            test_statistic=ts
    
    test_statistic -= model_selection.betabinlik(obs_Ers,obs_1_Ers,a0+a1-1,b0+b1-1).sum()
    #~ test_statistic -= sum(model_selection.betabinlik(Er,Nr,a=1.+sum(obs_Ers),b=1.+sum(obs_Nrs)).sum() for (Er,Nr) in izip(obs_Ers,obs_Nrs))
    print 'most likely change',best_change_point,test_statistic
    
    
    null_dist=np.zeros(n_samples)
    
    #NOTE: We assume that there is no overdispersion in the null distribution
    
    #calculate priors
    a_null=(1.+np.sum(obs_Ers,0)).flatten()
    b_null=(1.+np.sum(obs_1_Ers,0)).flatten()
    
    a0=(1.+np.sum(obs_Ers[:best_change_point],0)).flatten()
    a1=(1.+np.sum(obs_Ers[best_change_point:],0)).flatten()
    b0=(1.+np.sum(obs_1_Ers[:best_change_point],0)).flatten()
    b1=(1.+np.sum(obs_1_Ers[best_change_point:],0)).flatten()
    
    ts2=0
    
    # Generate all null edge counts
    Nrs = np.int32(a_null + b_null - 2)//w
    Ers = np.random.binomial(Nrs, a_null/(a_null + b_null), size=(n_samples,w,Nrs.size))
    _1_Ers = Nrs[np.newaxis,np.newaxis,:]-Ers
    
    # infer posterior
    a_null_ = (1.+np.sum(Ers,1))
    a0_=(1.+np.sum(Ers[:,:best_change_point,:],1))
    a1_=(1.+np.sum(Ers[:,best_change_point:,:],1))
    b_null_ = (1.+np.sum(_1_Ers,1))
    b0_=(1.+np.sum(_1_Ers[:,:best_change_point,:],1))
    b1_=(1.+np.sum(_1_Ers[:,best_change_point:,:],1))
    
    for i in xrange(w):
        
        if w<best_change_point:
            null_dist += model_selection.betabinlik(Ers[:,i,:],_1_Ers[:,i,:],a0_,b0_).sum(1)
            #~ ts2 += model_selection.betabinlik(obs_Ers[i].flatten(),obs_1_Ers[i].flatten(),a0,b0).sum()
        else:
            null_dist += model_selection.betabinlik(Ers[:,i,:],_1_Ers[:,i,:],a1_,b1_).sum(1)
            #~ ts2 += model_selection.betabinlik(obs_Ers[i].flatten(),obs_1_Ers[i].flatten(),a1,b1).sum()
        null_dist -= model_selection.betabinlik(Ers[:,i,:],_1_Ers[:,i,:],a_null_,b_null_).sum(1)
        #~ ts2 += model_selection.betabinlik(obs_Ers[i].flatten(),obs_1_Ers[i].flatten(),a_null,b_null).sum()
    
    p_value = 1-(test_statistic>null_dist).sum()/n_samples
    
    #~ print test_statistic, ts2
    
    print 'p value:',p_value
    
    
    return p_value, test_statistic, null_dist