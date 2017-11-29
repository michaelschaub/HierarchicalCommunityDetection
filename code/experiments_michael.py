from GHRGmodel import GHRG
import GHRGbuild
import spectral_algorithms as spectral
import metrics
import numpy as np
import scipy
from scipy import linalg
from matplotlib import pyplot as plt
import random
import sample_networks
import pickle

np.set_printoptions(precision=4,linewidth=200)

# Within ipython shell:
# %load_ext autoreload
# %autoreload 2
# %pylab
# import experiments_michael

def test_BHwithQR():
    n_nodes = 2**10
    n_groups = 8
    av_degree = 20
    n_levels =1
    SNR = 2
    a, b = GHRGbuild.calculateDegreesFromAvDegAndSNR(SNR,av_degree,n_groups)
    D_gen=GHRGbuild.create2paramGHRG(n_nodes,SNR,av_degree,n_levels,n_groups)
    partition_true = D_gen.get_partition_at_level(1)[0]
    G= D_gen.generateNetworkExactProb()
    A= D_gen.to_scipy_sparse_matrix(G)

    pvec = spectral.cluster_with_BetheHessian(A,num_groups=n_groups,mode='unweighted',
                                     regularizer='BHa',clustermode='qr')
    ol_score = metrics.overlap_score(pvec,partition_true)
    print ol_score

    pvec2= spectral.cluster_with_BetheHessian(A,num_groups=n_groups,mode='unweighted',
                                     regularizer='BHa',clustermode='kmeans')
    ol_score2 = metrics.overlap_score(pvec2,partition_true)
    print pvec2
    print ol_score2

    pvec3, _ = spectral.regularized_laplacian_spectral_clustering(A,num_groups=n_groups)
    ol_score3 = metrics.overlap_score(pvec3,partition_true)
    print ol_score3



def test_GHRG_hier(groups_per_level=4):
    # mean degree and number of nodes etc.
    n=1600
    n_levels = 2
    K=groups_per_level**n_levels
    av_degree = 30

    snr = 5

    a,b = GHRGbuild.calculateDegreesFromAvDegAndSNR(snr,av_degree,groups_per_level)
    D_gen=GHRGbuild.create2paramGHRG(n,snr,av_degree,n_levels,groups_per_level)

    G= D_gen.generateNetworkExactProb()
    A= D_gen.to_scipy_sparse_matrix(G)
    D = GHRG()
    D.infer_spectral_partition_hier(A)

    return D


def test_spectral_algorithms_hier_agg(n_levels=4,groups_per_level=2):
    random.seed(12345)
    # mean degree number of nodes etc.
    SNR = 9
    n=2**13
    K=groups_per_level**n_levels
    ratio = 0.1

    D_gen=create2paramGHRG(n,SNR,ratio,n_levels,groups_per_level)
    G=D_gen.generateNetworkExactProb()
    A = D_gen.to_scipy_sparse_matrix(G)
    # plt.figure()
    # plt.spy(A,markersize=0.5)

    pvec = spectral.spectral_partition(A,'Bethe',-1)
    # plt.figure()
    # plt.plot(pvec,marker='s')

    partition_true = D_gen.get_lowest_partition()
    partition_coarse = D_gen.partition_level(0)
    print partition_coarse

    ol_score = metrics.overlap_score(pvec,partition_true)
    print "Partition into "+ str(np.max(pvec)+1) +" groups"
    print "OVERLAP SCORE (SINGLE LAYER INFERENCE): ", ol_score, "\n\n"


    pvecs = spectral.hier_spectral_partition_agglomerate(A,pvec)
    pvecs = spectral.expand_partitions_to_full_graph(pvecs)
    print "RESULTS AGGLOMERATION PHASE"
    print pvecs

    ol_score = metrics.overlap_score(pvecs[0],partition_true)
    print "Partition into "+ str(np.max(pvecs[0])+1) +" groups"
    print "OVERLAP SCORE Finest: ", ol_score, "\n\n"

    ol_score = metrics.overlap_score(pvecs[-1],partition_coarse)
    print "Partition into "+ str(np.max(pvecs[-1])+1) +" groups"
    print "OVERLAP SCORE Coarsest: ", ol_score, "\n\n"


    return pvecs

"""
Experiment: Test Spectral inference algorithm on hierarchical test graph

Create a sequence of test graphs (realizations of a specified hier. random model) and try
to infer the true partition using spectral methods
"""
def test_spectral_algorithms_hier():
    # mean degree number of nodes etc.
    SNR = 10
    n=2**14

    # sample hier block model
    groups_per_level = np.array([2,2])
    A, pvecs_true = sample_networks.sample_hier_block_model(groups_per_level, av_deg = 15, nnodes=n, snr=SNR)

    # plt.figure()
    # plt.spy(A,markersize=1)
    print "Average degree"
    print A.sum(axis=1).mean()

    pvec = spectral.spectral_partition(A,'Bethe',-1)
    # plt.figure()
    # plt.plot(pvec,marker='s')

    ol_score = metrics.overlap_score(pvec,pvecs_true[-1])
    print "Partition into "+ str(np.max(pvec)+1) +" groups"
    print "OVERLAP SCORE (SINGLE LAYER INFERENCE): ", ol_score, "\n\n"


    print "\n\n Hier Partitioning\n"
    pvecs_inferred, _ = spectral.hier_spectral_partition(A)
    ol_score = metrics.calculate_level_comparison_matrix(pvecs_inferred,pvecs_true)
    print "\n\nOVERLAP SCORE\n", ol_score, "\n\n"


    return pvecs_inferred


def test_spectral_algorithms_non_hier():
    SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel = run_spectral_algorithms_non_hier()
    plot_results_overlap(SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel)

def test_spectral_algorithms_multiple_networks():
    SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel = run_spectral_algorithms_n_networks()
    plot_results_overlap(SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel)

def run_spectral_algorithms_n_networks(n_groups=4):
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio = 0.4

    SNR = np.arange(0.5,3,0.5)
    nsamples = 20
    overlap_Bethe = np.zeros((SNR.size,nsamples))
    overlap_Rohe = np.zeros((SNR.size,nsamples))
    overlap_Seidel = np.zeros((SNR.size,nsamples))
    nr_network_samples = 5

    for ii, snr in enumerate(SNR):

        # create GHRG object with specified parameters and create a sample network from it
        D_gen=create2paramGHRG(n,snr,ratio,n_levels,n_groups)
        partition_true = D_gen.get_lowest_partition()

        for jj in np.arange(nsamples):
            A = 0;
            for kk in xrange(nr_network_samples):
                G= D_gen.generateNetworkExactProb()
                A1= D_gen.to_scipy_sparse_matrix(G)
                A = A + A1

            A = A/float(nr_network_samples)


            pvec = spectral.spectral_partition(A,'Bethe',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Bethe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'Lap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Rohe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'SeidelLap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Seidel[ii,jj] = ol_score

    return SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel

def run_spectral_algorithms_non_hier(n_groups=4):
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio = 0.4

    SNR = np.arange(0.5,3,0.25)
    nsamples = 20
    overlap_Bethe = np.zeros((SNR.size,nsamples))
    overlap_Rohe = np.zeros((SNR.size,nsamples))
    overlap_Seidel = np.zeros((SNR.size,nsamples))

    for ii, snr in enumerate(SNR):

        # create GHRG object with specified parameters and create a sample network from it
        D_gen=create2paramGHRG(n,snr,ratio,n_levels,n_groups)
        partition_true = D_gen.get_lowest_partition()

        for jj in np.arange(nsamples):
            G= D_gen.generateNetworkExactProb()
            A= D_gen.to_scipy_sparse_matrix(G)


            pvec = spectral.spectral_partition(A,'Bethe',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Bethe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'Lap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Rohe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'SeidelLap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Seidel[ii,jj] = ol_score

    return SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel

def plot_results_overlap(SNR,overlap_Bethe,overlap_Rohe,overlap_Seidel):
    plt.figure()
    plt.errorbar(SNR, overlap_Bethe.mean(axis=1), overlap_Bethe.std(axis=1),label="BH")
    plt.errorbar(SNR, overlap_Rohe.mean(axis=1), overlap_Rohe.std(axis=1),label="regL")
    plt.errorbar(SNR, overlap_Seidel.mean(axis=1), overlap_Seidel.std(axis=1),label="SL")
    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel("overlap score")
