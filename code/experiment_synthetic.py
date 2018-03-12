from __future__ import division
import numpy as np
import GHRGbuild
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
plt.ion()

def complete_inf(symmetric=True):
    
    groups_per_level=3
    n_levels=3
    
    n=3**9
    
    c_bar=50
    
    for rep in xrange(20):
    
        for snr in np.arange(0.5,10.5,0.5):
            
            print 'SNR',snr
            
            if symmetric:
                D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
                
            else :
                pass
            
            #generate graph and create adjacency
            G=D_actual.generateNetworkExactProb()
            A=D_actual.to_scipy_sparse_matrix(G)
            #get true hierarchy
            true_pvec = D_actual.get_partition_all()

            #infer partitions with no noise
            inf_pvec = spectral.hier_spectral_partition(A, reps=20)
            
            #calculate scores
            score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, true_pvec)
            precision, recall = metrics.calculate_precision_recall(score_matrix)
            diff_levels = metrics.compare_levels(true_pvec,inf_pvec)
            bottom_lvl = score_matrix[-1,-1]
            print "\n\nRESULTS\n\nbottom level"
            print bottom_lvl
            print len(inf_pvec), len(true_pvec)
            print diff_levels
            print "precision, recall"
            print precision, recall
            
            
            print [len(np.unique(pv)) for pv in true_pvec]
            print [len(np.unique(pv)) for pv in inf_pvec]
            
            with open('results/complete_inf_{}.txt'.format({True : 'sym', False : 'asym'}[symmetric]),'a') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr,precision,recall,bottom_lvl,len(inf_pvec)))
                for lvl in inf_pvec:
                    file.write(' {}'.format(len(np.unique(lvl))))
                file.write('\n')



def plot_complete(symmetric=True):
    
    with open('results/complete_inf_{}.txt'.format({True : 'sym', False : 'asym'}[symmetric])) as file:
        results = file.readlines()
    
    result
    
    plt.figure()
    
    