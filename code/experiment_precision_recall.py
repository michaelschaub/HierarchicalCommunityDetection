#!/usr/bin/env python
from __future__ import division
import GHRGmodel
import GHRGbuild
import metrics
import numpy as np
from matplotlib import pyplot as plt



def loop_over_hierarchical_clustering(n_levels = 2, groups_per_level =2 , n= 2**13):

    SNR_max = 1
    SNR_min = -1
    SNR_step = .2
    SNR_range = 10**np.arange(SNR_min,SNR_max+SNR_step,SNR_step)
    av_degree = 12
    nsamples = 20

    precision = np.zeros((nsamples,SNR_range.shape[0]))
    recall = np.zeros((nsamples,SNR_range.shape[0]))

    for j, SNR in enumerate(SNR_range):
        print "SNR =", SNR, "\n"
        D_gen=GHRGbuild.create2paramGHRG(n,SNR,av_degree,n_levels,groups_per_level)

        # retrieve planted partition structure
        pvecs_true = D_gen.get_partition_all()
        print "\nTRUE VECTORS"
        print pvecs_true

        # sample networks and compare with inferred
        for i in np.arange(nsamples):

            # sample network adjacency
            G= D_gen.generateNetworkExactProb()
            A= D_gen.to_scipy_sparse_matrix(G)

            D_inf = GHRGmodel.GHRG()
            D_inf.infer_spectral_partition_hier(A)
            pvecs_inferred = D_inf.get_partition_all()
            print "\nINFERRED PARTITION"
            print pvecs_inferred

            ol_matrix = metrics.calculate_level_comparison_matrix(pvecs_inferred,pvecs_true)
            print "\nOVERLAP MATRIX"
            print ol_matrix

            alignment = (pvecs_inferred[0] == pvecs_true[0]).sum()
            print "\nRaw alignment"
            print alignment/n

            p,r=metrics.calculate_precision_recall(ol_matrix)
            precision[i,j] = p
            recall[i,j] = r

            print "\n precision and recall"
            print p
            print r

    plt.figure()
    plt.plot(SNR_range,precision.mean(axis=0),'b-.')
    plt.plot(SNR_range,recall.mean(axis=0),'r--')
