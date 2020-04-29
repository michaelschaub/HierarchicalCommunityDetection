#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:


"""
import numpy as np
from scipy import sparse
from cluster import Partition, Hierarchy
from error_handling import RecursionLimitError


def create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level):
    """
    Function to create a test GHRG for simulations.
    Parameters:
        n   : number of nodes
        snr : signal to noise ratio (1 represents theoretical detectability
              threshold)
        c_bar : average degree
        n_levels    : depth of GHRG
        groups_per_level     : number of groups at each level
    """

    dendro = HierarchicalGraph()
    omega = []
    partitions = []
    n_this_level = n
    for level in range(1, n_levels):
        print("Hierarchy Level: ", level)
        # calculate omega for this level
        omega = calculate2paramOmega(n_this_level, snr, c_bar,
                                     groups_per_level)

        # update n and c_bar
        cin = omega[0, 0]*n_this_level
        n_this_level = n_this_level / groups_per_level
        if np.floor(n_this_level) != n_this_level:
            print("Rounding number of nodes")

        c_bar = (cin/n_this_level)*(n_this_level / groups_per_level)

    # for level in range(1, n_levels):
        num_current_groups = groups_per_level**(level)
        pvec = np.kron(np.arange(num_current_groups, dtype=int),
                       np.ones(int(groups_per_level), dtype=int))
        part = PlantedPartition(pvec, omega)
        dendro.add_level(part)

    # complete finest level
    print("Hierarchy Level: ", n_levels)
    num_current_groups = groups_per_level**(n_levels)
    omega = calculate2paramOmega(n_this_level, snr, c_bar,
                                 groups_per_level)
    pvec = np.kron(np.arange(num_current_groups, dtype=int),
                   np.ones(int(n/num_current_groups), dtype=int))
    part = PlantedPartition(pvec, omega)
    dendro.add_level(part)
    partitions = partitions[::-1]

    return dendro
    # matrix = omega[0]
    # for level in range(n_levels-1):
    #     Omega = matrix_fill_in_diag_block(omega[level+1],matrix)
    #     matrix = Omega
    #
    # # Q: should we adjust the Hierarchy constructors to make this less cumbersome
    # Hier = Hierarchy(Partition(partitions.pop(0)))
    # for pvec in partitions:
    #     Hier.add_level(Partition(pvec))
    #
    # graph = HierarchicalGraph(Hier, Omega)

    # return graph


class HierarchicalGraph(Hierarchy):

    def __init__(self):
        pass

    def calc_nodes_per_level(self):
        # calculate group sizes at coarsest level
        self[-1].group_size = np.ravel(self[-1].H.sum(0).astype(int))
        lastH = self[-1].H
        # calculate group sizes at the rest of the levels
        for part1, part2 in zip(self[-1:0:-1], self[-2::-1]):

            lastH = lastH @ part2.H
            part2.group_size = np.ravel(lastH.sum(0).astype(int))

        # calculate total number of nodes
        self.n = self[0].group_size.sum()

    def sample_edges_in_block(self, level, i, j):
        partition = self[level]
        omega = partition.omega
        gs = partition.group_size
        rg = np.random.default_rng()

        pij = omega[i, j]
        gsi = gs[i]
        gsj = gs[j]
        n_nodes = gsi*gsj
        offseti = gs[:i].sum()
        offsetj = gs[:j].sum()
        # sample number of edges
        n_edges = rg.binomial(n_nodes, pij)
        # sample which edges in the block
        edge_idx = rg.choice(n_nodes, n_edges, replace=False)
        source_nodes = [offseti + (ei // gsi) for ei in edge_idx]
        target_nodes = [offsetj + (ei % gsi) for ei in edge_idx]
        return source_nodes, target_nodes

    def sample_edges_at_level(self, level):
        source_nodes = []
        target_nodes = []
        try:
            partition = self[level]
            n_groups = len(partition.omega)
            for i, j in zip(*np.triu_indices(n_groups, 1)):
                tmp_s, tmp_t = self.sample_edges_in_block(level, i, j)
                source_nodes.extend(tmp_s)
                target_nodes.extend(tmp_t)

            partition = self[level]
            gs = partition.group_size

            for i in range(n_groups):
                offseti = gs[:i].sum()
                try:
                    tmp_s, tmp_t = self.sample_edges_at_level(level+1)
                    source_nodes.extend([s+offseti for s in tmp_s])
                    target_nodes.extend([t+offseti for t in tmp_t])
                except RecursionLimitError:
                    tmp_s, tmp_t = self.sample_edges_in_block(level, i, i)
                    tmp_s, tmp_t = zip(*[(s, t) for s, t in zip(tmp_s, tmp_t)
                                       if t > s])
                    source_nodes.extend(tmp_s)
                    target_nodes.extend(tmp_t)

        # end of recursion
        except IndexError:
            raise RecursionLimitError('recursion end')

        return source_nodes, target_nodes

    def sample_network(self):
        self.calc_nodes_per_level()
        n = self.n
        source_nodes, target_nodes = self.sample_edges_at_level(0)

        A = sparse.coo_matrix((np.ones(len(source_nodes)),
                              (source_nodes, target_nodes)),
                              shape=(n, n)).tocsr()
        # reciprocate links
        A = A + A.T
        return A


class PlantedPartition(Partition):

    def __init__(self, pvec, omega):
        self.pvec = pvec
        self.omega = omega
        self.relabel_partition_vec()
        self.create_partition_matrix()
        self.nc = np.asarray(self.H.sum(0).astype(int))
        # self.nc = np.array([sum(self.pvec == i) for i in range(len(self))])


def calculate2paramOmega(n, snr, c_bar, groups_per_level):

    cin, cout = calculateDegreesFromAvDegAndSNR(snr, c_bar,
                                                groups_per_level)
    print('KS Detectable: ', snr >= 1,
          "| Link Probabilities in / out per block: ", cin/n,
          cout/n)
    # Omega is assigned on a block level, i.e. for each level we have one
    # omega array
    # this assumes a perfect hierarchy with equal depth everywhere

    omega = np.eye(groups_per_level)
    omega *= (cin/n-cout/n)
    omega += cout/n

    if np.any(omega >= 1):
        print("no probability > 1 not allowed")
        raise ValueError("Something wrong")

    return omega


def calculateDegreesFromAvDegAndSNR(SNR, av_degree, num_cluster=2):
    """
    Given a particular signal to noise ratio (SNR), the average degree and a
    number of clusters,
    compute the degree parameters for a planted partition model.

    Output:  degree parameters a, b, such that the probability for an 'inside'
    connection is a/n, and for an outside connection b/n.
    """
    amb = num_cluster * np.sqrt(av_degree*SNR)
    b = av_degree - amb/float(num_cluster)
    a = amb + b

    return a, b
