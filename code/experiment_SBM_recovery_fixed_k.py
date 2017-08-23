#!/usr/bin/env python
from __future__ import division
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
import metrics
import scipy
import numpy as np
from matplotlib import pyplot as plt
import graph_tool as gt
from graph_tool import inference

def run_algorithm_comparison(n_nodes=2**12):

    # loop over particular SNR regime setup
    log_SNR_min = -0.5
    log_SNR_max = 0.5
    log_SNR_step = 0.05
    SNR = 10**np.arange(log_SNR_min,log_SNR_max+log_SNR_step,log_SNR_step)

    # some further statistics for the network, we have just one hier. level here
    av_degree = 20
    n_samples = 20
    n_levels = 1
    n_groups = 32

    # preallocate results
    overlap_Bethe = np.zeros((SNR.size,n_samples))
    overlap_Rohe = np.zeros((SNR.size,n_samples))
    overlap_Tiago = np.zeros((SNR.size,n_samples))

    for ii, snr in enumerate(SNR):

        # build SBM according to the specification
        a, b = GHRGbuild.calculateDegreesFromAvDegAndSNR(snr,av_degree,n_groups)
        D_gen=GHRGbuild.create2paramGHRG(n_nodes,snr,av_degree,n_levels,n_groups)
        partition_true = D_gen.get_partition_at_level(1)[0]

        for jj in np.arange(n_samples):
            # Generate samples from the model
            G= D_gen.generateNetworkExactProb()
            A= D_gen.to_scipy_sparse_matrix(G)
            GT_graph = gt.Graph()
            #TODO: double check the creation of the GT graphs is correct
            GT_graph.add_edge_list(scipy.transpose(A.nonzero()))

            # Spectral Techniques
            pvec = spectral.spectral_partition(A,'Bethe',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Bethe[ii,jj] = ol_score
            print "1"

            pvec = spectral.spectral_partition(A,'Lap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Rohe[ii,jj] = ol_score
            print "2"

            # Inference based on Tiago
            blockmodel_state = gt.inference.minimize_blockmodel_dl(GT_graph,
                                                                   B_min=n_groups,
                                                                   B_max=n_groups,
                                                                   deg_corr=False)
            pvec = blockmodel_state.get_blocks().get_array()
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Tiago[ii,jj] = ol_score
            print "3"



    plot_results_overlap(SNR, overlap_Bethe, overlap_Rohe, overlap_Tiago)
    plt.savefig('fixed_k.pdf', bbox_inches='tight')

def plot_results_overlap(SNR,overlap_Bethe,overlap_Rohe,overlap_Tiago):
    plt.figure()
    plt.errorbar(SNR, overlap_Bethe.mean(axis=1), overlap_Bethe.std(axis=1),label="BH")
    plt.errorbar(SNR, overlap_Rohe.mean(axis=1), overlap_Rohe.std(axis=1),label="regL")
    plt.errorbar(SNR, overlap_Tiago.mean(axis=1), overlap_Tiago.std(axis=1),label="Tiago")
    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel("overlap score")
    plt.show()
