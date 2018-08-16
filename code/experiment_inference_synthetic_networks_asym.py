from __future__ import division
import numpy as np
import GHRGbuild
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
plt.ion()

def complete_inf(groups_per_level=3, n_levels=3,prefix="results"):

    n=3**9

    c_bar=50

    for rep in xrange(50):

        for snr in np.arange(0.5,10.5,0.5):

            print 'SNR',snr
            D_actual=GHRGbuild.createAsymGHRG(n,snr,c_bar,n_levels,groups_per_level)


            #generate graph and create adjacency
            G=D_actual.generateNetworkExactProb()
            A=D_actual.to_scipy_sparse_matrix(G)
            #get true hierarchy
            true_pvec = D_actual.get_partition_all()
            #~ plt.figure()
            #~ plt.spy(A,markersize=2e-2)
            #~ print A.nnz
            #~ for tp in true_pvec:
                #~ plt.plot(tp)
                #~ print np.unique(tp)

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

            with open('results_asym/{}_complete_inf_{}_{}.txt'.format(prefix, n_levels, groups_per_level),'a+') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr,precision,recall,bottom_lvl,len(inf_pvec)))
                for lvl in inf_pvec:
                    file.write(' {}'.format(len(np.unique(lvl))))
                file.write('\n')



def infer_k_known(groups_per_level=3, n_levels=3, model='SBM', prefix="results"):

    n = 3**9

    c_bar = 50

    for rep in xrange(50):
    # for rep in xrange(10):

        for snr in np.arange(0.5, 10.5, 0.5):
            # for snr in np.arange(0.5,10.5,1):

            print 'SNR', snr

            D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar,
                                                      n_levels, groups_per_level)


            # generate graph and create adjacency
            G = D_actual.generateNetworkExactProb()
            A = D_actual.to_scipy_sparse_matrix(G)
            # get true hierarchy
            true_pvec = D_actual.get_partition_all()

            Ks = np.array([len(np.unique(pv)) for pv in true_pvec])[::-1]
            print 'Ks', Ks

            # infer partitions with no noise
            inf_pvec = spectral.hier_spectral_partition(A, Ks=Ks, model=model)

            # calculate scores
            score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, true_pvec)
            precision, recall = metrics.calculate_precision_recall(score_matrix)
            diff_levels = metrics.compare_levels(true_pvec, inf_pvec)
            bottom_lvl = score_matrix[-1, -1]
            print "\n\nRESULTS\n\nbottom level"
            print bottom_lvl
            print len(inf_pvec), len(true_pvec)
            print diff_levels
            print "precision, recall"
            print precision, recall

            print [len(np.unique(pv)) for pv in true_pvec]
            print [len(np.unique(pv)) for pv in inf_pvec]
            print "\n\nEND RESULTS\n\n"

            with open('results_asym/{}_knownK_inf_{}_{}.txt'.format(prefix, n_levels, groups_per_level), 'a+') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr, precision, recall, bottom_lvl, len(inf_pvec)))
                for lvl in xrange(len(inf_pvec)):
                    file.write(' {:.3f}'.format(score_matrix[lvl, lvl]))
                file.write('\n')


def infer_agglomeration(symmetric=True, groups_per_level=3, n_levels=3,prefix="results"):

    n=3**9

    c_bar=50

    for rep in xrange(50):

        for snr in np.arange(0.5,10.5,0.5):

            print 'SNR',snr

            D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)


            #generate graph and create adjacency
            G=D_actual.generateNetworkExactProb()
            A=D_actual.to_scipy_sparse_matrix(G)
            #get true hierarchy
            true_pvec = D_actual.get_partition_all()

            #infer partitions with no noise
            inf_pvec = hier_spectral_partition_agglomerate(A,true_pvec)

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

            with open('results_asym/{}_agglomerate_inf_{}_{}.txt'.format(prefix, n_levels, groups_per_level),'a+') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr,precision,recall,bottom_lvl,len(inf_pvec)))
                for lvl in inf_pvec:
                    file.write(' {}'.format(len(np.unique(lvl))))
                file.write('\n')

def plot_levels(symmetric=True, groups_per_level=3, n_levels=3, prefix="results"):
    # ~ groups_per_level=3
    # ~ n_levels=3
    with open('results_asym/{}_knownK_inf_{}_{}.txt'.format(prefix, n_levels, groups_per_level)) as file:
        results = file.readlines()

    scores = np.float64([result.strip().replace('*', '').split() for result in results])

    snrs = np.unique(scores[:,0])
    n = len(snrs)
    precision = np.empty(n)
    recall = np.empty(n)
    overlap1 = np.empty(n)
    overlap2 = np.empty(n)
    overlap3 = np.empty(n)
    levels = np.empty(n)

    for si, snr in enumerate(snrs):
        idxs = scores[:, 0] == snr
        # ~ print snr,idxs, (scores[idxs,1])
        precision[si] = np.mean(scores[idxs, 1])
        recall[si] = np.mean(scores[idxs, 2])
        overlap1[si] = np.mean(scores[idxs, 5])
        overlap2[si] = np.mean(scores[idxs, 6])
        overlap3[si] = np.mean(scores[idxs, 7])
        levels[si] = np.mean(scores[idxs, 4])

    # plot level overlap
    plt.figure()
    plt.plot(snrs, overlap1, label='overlap 1')
    plt.plot(snrs, overlap2, label='overlap 2')
    plt.plot(snrs, overlap3, label='overlap 3')
    plt.axvline(1, ls=':', color='k', lw=0.5)
    plt.axhline(0, color='k', lw=0.5)
    plt.legend()
    plt.tight_layout()



def plot_complete(groups_per_level=3, n_levels=3,prefix="results"):
    #~ groups_per_level=3
    #~ n_levels=3
    with open('results_asym/{}_complete_inf_{}_{}.txt'.format(prefix, n_levels, groups_per_level)) as file:
        results = file.readlines()

    scores = np.float64([result.split('*')[0].split() for result in results])

    snrs = np.unique(scores[:,0])
    n = len(snrs)
    precision = np.empty(n)
    recall = np.empty(n)
    overlap = np.empty(n)
    levels = np.empty(n)

    for si,snr in enumerate(snrs):
        idxs = scores[:,0]==snr
        #~ print snr,idxs, (scores[idxs,1])
        precision[si] = np.mean(scores[idxs,1])
        recall[si] = np.mean(scores[idxs,2])
        overlap[si] = np.mean(scores[idxs,3])
        levels[si] = np.mean(scores[idxs,4])

    #plot precision recall
    plt.figure()
    plt.plot(snrs, precision, label='precision')
    plt.plot(snrs, recall, label='recall')
    plt.axvline(1,ls=':',color='k',lw=0.5)
    plt.axhline(0,color='k',lw=0.5)
    plt.legend()
    plt.tight_layout()

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(snrs, overlap, label='overlap',color=color)
    ax1.set_ylabel('overlap',color=color)
    ax1.set_xlabel('SNR')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.plot(snrs, levels, '*', label='levels',color=color)
    ax2.set_ylabel('# levels',color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.axvline(1,ls=':',color='k',lw=0.5)
    plt.axhline(0,color='k',lw=0.5)

    plt.tight_layout()



