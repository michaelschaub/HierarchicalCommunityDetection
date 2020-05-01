import numpy as np
import metrics
from inference import infer_hierarchy
from generation import create2paramGHRG, createAsymGHRG
from matplotlib import pyplot as plt
import time

def complete_inf_sym(groups_per_level=3, n_levels=3, prefix="results"):

    n = 3**9

    c_bar = 50

    for rep in range(50):

        for snr in np.arange(0.5, 10.5, 0.5):

            print('\n\nSNR', snr)

            Hgraph = create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
            print("Model generated")

            # generate adjacency
            tic = time.perf_counter()
            A = Hgraph.sample_network()
            print("adjacency matrix sampled")
            toc = time.perf_counter()
            print(f"Sample in {toc - tic:0.4f} seconds")

            # infer partitions 
            inf_pvec = infer_hierarchy(A)

            # calculate scores
            score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, Hgraph)
            precision, recall = metrics.calculate_precision_recall(score_matrix)
            diff_levels = metrics.compare_levels(Hgraph, inf_pvec)
            bottom_lvl = score_matrix[-1, -1]
            print("\n\nRESULTS\n\nbottom level")
            print(bottom_lvl)
            print(len(inf_pvec), len(Hgraph))
            print(diff_levels)
            print("precision, recall")
            print(precision, recall)

            print([pv.k for pv in Hgraph])
            print([pv.k for pv in inf_pvec])

            with open('results/{}_complete_inf_{}_{}_{}.txt'.format(prefix, 'sym', n_levels, groups_per_level), 'a+') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr, precision, recall, bottom_lvl, len(inf_pvec)))
                for lvl in inf_pvec:
                    file.write(' {}'.format(lvl.k))
                file.write('\n')

def complete_inf_asym(groups_per_level=3, n_levels=3, prefix="results"):

    n = 3**9

    c_bar = 50

    for rep in range(50):

        for snr in np.arange(0.5, 10.5, 0.5):

            print('\n\nSNR', snr)

            Hgraph = createAsymGHRG(n, snr, c_bar, n_levels, groups_per_level)
            print("Model generated")

            # generate adjacency
            tic = time.perf_counter()
            A = Hgraph.sample_network()
            print("adjacency matrix sampled")
            toc = time.perf_counter()
            print(f"Sample in {toc - tic:0.4f} seconds")

            # infer partitions 
            inf_pvec = infer_hierarchy(A)

            # calculate scores
            score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, Hgraph)
            precision, recall = metrics.calculate_precision_recall(score_matrix)
            diff_levels = metrics.compare_levels(Hgraph, inf_pvec)
            bottom_lvl = score_matrix[-1, -1]
            print("\n\nRESULTS\n\nbottom level")
            print(bottom_lvl)
            print(len(inf_pvec), len(Hgraph))
            print(diff_levels)
            print("precision, recall")
            print(precision, recall)

            print([pv.k for pv in Hgraph])
            print([pv.k for pv in inf_pvec])

            with open('results/{}_complete_inf_{}_{}_{}.txt'.format(prefix, 'asym', n_levels, groups_per_level), 'a+') as file:
                file.write('{} {:.3f} {:.3f} {:.3f} {} *'.format(snr, precision, recall, bottom_lvl, len(inf_pvec)))
                for lvl in inf_pvec:
                    file.write(' {}'.format(lvl.k))
                file.write('\n')