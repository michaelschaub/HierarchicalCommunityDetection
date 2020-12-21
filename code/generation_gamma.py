#!/usr/bin/env python
"""
Spectral clustering functions for hier clustering.
The module contains:


"""
import numpy as np
from scipy import sparse
from cluster import Partition, Hierarchy
from error_handling import RecursionLimitError


def create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level, gamma=0.9):
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
    n_this_level = n
    num_current_groups = 1
    for level in range(1, n_levels):
        print("Hierarchy Level: ", level)
        # calculate omega for this level
        omega = calculate2paramOmega(n, snr, c_bar,
                                     groups_per_level, gamma, level)
        # create omega_map (each group in the previous level has an omega)
        omega_map = dict((i, 0) for i in range(num_current_groups))
        # update n and c_bar
        # cin = omega[0, 0]*n_this_level
        n_this_level = n_this_level / groups_per_level
        if np.floor(n_this_level) != n_this_level:
            print("Rounding number of nodes")

        # c_bar = (cin/n_this_level)*(n_this_level / groups_per_level)

        num_current_groups = groups_per_level**(level)
        pvec = np.kron(np.arange(num_current_groups, dtype=int),
                       np.ones(int(groups_per_level), dtype=int))
        part = PlantedPartition(pvec, [omega], omega_map)
        dendro.add_level(part)

    # complete finest level
    print("Hierarchy Level: ", n_levels)
    omega = calculate2paramOmega(n, snr, c_bar,
                                 groups_per_level, gamma, n_levels)
    # create omega_map (each group in the previous level has an omega)
    omega_map = dict((i, 0) for i in range(num_current_groups))
    num_current_groups = groups_per_level**(n_levels)
    pvec = np.kron(np.arange(num_current_groups, dtype=int),
                   np.ones(int(n/num_current_groups), dtype=int))
    part = PlantedPartition(pvec, [omega], omega_map)
    dendro.add_level(part)
    dendro.expand_partitions_to_full_graph()

    return dendro


def createAsymGHRG(n, snr, c_bar, n_levels, groups_per_level, gamma=0.9):
    """
    Function to create an asymmetric test GHRG for simulations
    Parameters:
        n   : number of nodes
        snr : signal to noise ratio (1 represents theoretical detectability threshold)
        c_bar : average degree
        n_levels    : depth of GHRG
        groups_per_level     : number of groups at each level
    """

    dendro = HierarchicalGraph()
    n_this_level = n
    num_current_groups = 1
    pvec_final = np.zeros(n)
    for level in range(1, n_levels):
        print("Hierarchy Level: ", level)
        # calculate omega for this level
        omega = calculate2paramOmega(n, snr, c_bar,
                                     groups_per_level, gamma, level)
        # create omega_map (each group in the previous level has an omega)
        omega_map = {num_current_groups-1: 0}
        # update pvec_final
        new_groups = np.arange(groups_per_level, dtype=int) + pvec_final[-1]
        n_each_group = int(n_this_level/groups_per_level)
        pvec_final[-n_this_level:] = np.kron(new_groups,
                                             np.ones(n_each_group, dtype=int))
        # update n and c_bar
        # cin = omega[0, 0]*n_this_level
        n_this_level = int(n_this_level / groups_per_level)
        if n_this_level % groups_per_level > 0:
            print("Rounding number of nodes")

        # c_bar = (cin/n_this_level)*(n_this_level / groups_per_level)

        # update number of current groups
        num_current_groups += groups_per_level-1
        # print(num_current_groups, num_current_groups + groups_per_level-1)
        # initialise pvec
        pvec = np.empty(num_current_groups + groups_per_level-1)
        pvec[:num_current_groups] = np.arange(num_current_groups)
        pvec[num_current_groups:] = num_current_groups-1

        part = PlantedPartition(pvec, [omega], omega_map)
        dendro.add_level(part)

    # complete finest level
    print("Hierarchy Level: ", n_levels)
    omega = calculate2paramOmega(n, snr, c_bar,
                                 groups_per_level, gamma, n_levels)
    omega_map = {num_current_groups-1: 0}
    # update pvec_final
    new_groups = np.arange(groups_per_level, dtype=int) + pvec_final[-1]
    n_each_group = int(n_this_level/groups_per_level)
    pvec_final[-n_this_level:] = np.kron(new_groups,
                                         np.ones(n_each_group, dtype=int))
    part = PlantedPartition(pvec_final, [omega], omega_map)
    dendro.add_level(part)
    dendro.expand_partitions_to_full_graph()

    return dendro


class HierarchicalGraph(Hierarchy):

    def __init__(self):
        pass

    def count_links_between_groups(self, A, partition_idx, directed=True,
                                   self_loops=False):
        """
        Compute the number of possible and actual links between the groups
        indicated in the partition vector.
        """
        partition = self[partition_idx]
        # H = partition.H
        lastH = self[-1].H
        for partition in self[::-1][1:len(self)-partition_idx]:
            lastH = lastH @ partition.H
        # print(lastH.shape, H.shape)
        # lastH = lastH @ H
        nodes_per_group = np.ravel(lastH.sum(0))

        # each block counts the number of half links / directed links
        links_between_groups = (lastH.T @ A @ lastH)
        # convert to dense matrix (if sparse, otherwise continue)
        try:
            links_between_groups = links_between_groups.A
        except AttributeError:
            pass

        # convert to array type first, before performing outer product
        possible_links = np.outer(nodes_per_group, nodes_per_group)

        # if we do not allow self-loops this needs adjustment.
        if not self_loops:
            possible_links = possible_links - np.diag(nodes_per_group)

        if not directed:
            # we need to scale diagonal only by factor 2
            links_between_groups -= np.diag(np.diag(links_between_groups)) / 2
            links_between_groups = np.triu(links_between_groups)

            possible_links -= np.diag(np.diag(possible_links)) / 2
            possible_links = np.triu(possible_links)

        return links_between_groups, possible_links

    def expand_partitions_to_full_graph(self):
        """
        Map list of aggregated partition vectors in heirarchy to list of
        full-sized partition vectors
        """
        # the finest partition is already at the required size
        self[-1].pvec_expanded = self[-1].pvec

        # loop over all other partition
        for p0, partition in zip(self[:0:-1], self[-2::-1]):
            # group indices of previous level correspond to nodes in the
            # aggregated graph;
            # get the group ids of those nodes, and expand by reading out one
            # index per previous node
            partition.pvec_expanded = partition.pvec[p0.pvec_expanded]

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

        # calculate group sizes by branch
        self.tree_dict = {}
        self.tree_dict[0] = {0: np.arange(self[0].k)}
        for level in range(1, len(self)):
            self.tree_dict[level] = {}
            for b in range(self[level-1].k):
                self.tree_dict[level][b], _ = np.nonzero(self[level-1].H[:, b])

    def sample_edges_in_block(self, level, branch, i, j):
        """
        Sample edges within a specified block.
        level :  level in the dendrogram
        branch : branch number at level
        i, j : block indices (relative to branch)
        """
        # get local variables: partition, omega etc.
        partition = self[level]
        tree_dict = self.tree_dict
        omega_idx = partition.omega_map[branch]
        omega = partition.omega_list[omega_idx]
        gs = partition.group_size

        # initialise random number generator
        rg = np.random.default_rng()

        pij = omega[i, j]
        groups = self.tree_dict[level][branch]
        # get true block indices for the level (i and j are relative to branch)
        truei = groups[i]
        truej = groups[j]
        # get no of nodes of both groups
        gsi = gs[truei]
        gsj = gs[truej]
        # possible number of edges
        block_size = gsi*gsj
        # calculate global offsets
        offseti = gs[:tree_dict[level][branch][i]].sum()
        offsetj = gs[:tree_dict[level][branch][j]].sum()

        # sample number of edges
        n_edges = rg.binomial(block_size, pij)
        # sample which edges in the block
        edge_idx = rg.choice(block_size, n_edges, replace=False)
        source_nodes = [offseti + (ei // gsi) for ei in edge_idx]
        target_nodes = [offsetj + (ei % gsi) for ei in edge_idx]
        return source_nodes, target_nodes

    def sample_edges_at_level(self, level, branch):
        """
        Sample edges starting at a given level (and branch).
        Recursive function to traverse all nodes in the hierarchy below
        specifies level and branch.
        """
        # initialise lists for source and target nodes.
        source_nodes = []
        target_nodes = []

        try:
            # KeyError will be raised here if lowest level has been reached
            groups = self.tree_dict[level][branch]
            n_groups = len(groups)
            # first sample edges in the off-diagonal blocks at current level
            for i, j in zip(*np.triu_indices(n_groups, 1)):
                tmp_s, tmp_t = self.sample_edges_in_block(level, branch, i, j)
                source_nodes.extend(tmp_s)
                target_nodes.extend(tmp_t)

            # sample edges in the diagonal blocks
            for i in range(len(groups)):
                try:
                    # try to go deeper
                    tmp_s, tmp_t = self.sample_edges_at_level(level+1,
                                                              groups[i])
                    source_nodes.extend(tmp_s)
                    target_nodes.extend(tmp_t)

                except RecursionLimitError:
                    # when final depth reached sample edges using the diagonal
                    # of omega at the current branch
                    tmp_s, tmp_t = self.sample_edges_in_block(level, branch,
                                                              i, i)
                    # remove 'lower triangle' because undirected
                    tmp_s, tmp_t = zip(*[(s, t) for s, t in zip(tmp_s, tmp_t)
                                       if t > s])
                    source_nodes.extend(tmp_s)
                    target_nodes.extend(tmp_t)

        # end of recursion
        except KeyError:
            raise RecursionLimitError('recursion end')

        return source_nodes, target_nodes

    def sample_network(self):
        """
        Sample a network.
        Samples an undirected network from the Planted Partitions in the
        hierarchy. Currently assumes hierarchy is simple assortative.
        """
        self.calc_nodes_per_level()
        n = self.n
        source_nodes, target_nodes = self.sample_edges_at_level(0, 0)

        A = sparse.coo_matrix((np.ones(len(source_nodes)),
                              (source_nodes, target_nodes)),
                              shape=(n, n)).tocsr()
        # reciprocate links
        A = A + A.T
        return A


class PlantedPartition(Partition):

    def __init__(self, pvec, omega_list, omega_map):
        self.pvec = pvec
        self.omega_list = omega_list
        self.omega_map = omega_map
        self.relabel_partition_vec()
        self.create_partition_matrix()
        self.nc = np.asarray(self.H.sum(0).astype(int))


def calculate2paramOmega(n, snr, c_bar, groups_per_level, gamma, level=1):
    couts = []
    for l in range(1, level+1):
        snr_lvl = snr * gamma ** (l-1)
        cin, cout_ = calculateDegreesFromAvDegAndSNR(snr_lvl, c_bar,
                                                     groups_per_level**l)
        # print('cin/cout', cin, cout_)
        cout = cout_ * (groups_per_level**l - 1)
        # print(f'cout_*{groups_per_level}^{l}', cout)
        for lvl, coutl in enumerate(couts[::-1], start=1):
            cout -= coutl * (groups_per_level-1) * groups_per_level**lvl
            # print(f'cout {lvl}: {coutl} * (k-1)k^{lvl} = {cout}')
        cout /= groups_per_level - 1
        # print('final', cout)
        couts.append(cout)
    print('KS Detectable: ', snr >= 1,
          "| Link Probabilities in / out per block: ", cin/n,
          cout/n)
    print(f'Number of nodes: {n} | In / out degree: {cin} / {cout}')
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
