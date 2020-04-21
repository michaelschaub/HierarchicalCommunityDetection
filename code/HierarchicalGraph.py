#!/usr/bin/env python
import numpy as np
import scipy.sparse
from generate_hier_graphs import sample_block_model

from cluster import Hierarchy
from cluster import Partition

# TODO is this the right abstraction? it seems we are not using the list inheritence properly?!
class HierarchicalGraph(Hierarchy):

    ############################
    ## PART 1 -- CONSTRUCTOR ETC
    ############################
    def __init__(self, hierarchy=Hierarchy(Partition([0])), omega=0):
        self.PlantedHier = hierarchy
        self.Omega = omega
        if len(self.PlantedHier[-1]) != self.Omega.shape[0]:
            raise NotImplementedError("Finest partition has to match dimensions of affinity matrix")


    ################################
    ## PART 2 --- OUTPUT INFORMATION
    ################################

    def get_partition_at_level(self,level):
        """Return the partition at a specified level of the dendrogram
           level == 1 corresponds to coarsest (non-trivial) partition detected
           level == -1 corresponds to finest (non-trival) partition detected
        """
        return self.PlantedHier[level]

    def get_number_of_levels(self):
        return len(self.PlantedHier)

    def get_partition_all(self):
        return self.PlantedHier

    def print_partition_at_level(self,level):
        print(self.get_partition_at_level(level).pvec)

    def print_partition_all(self):
        for level in range(len(self.PlantedHier)):
            print(self.get_partition_at_level(level).pvec)

    def checkDetectabliityGeneralSBM(self):
        """
        Compute the SNR for a general Omega 1 
        """
        pvec = self.get_partition_at_level(-1).pvec
        nc = [sum(pvec == i) for i in range(pvec.max() + 1)]

        # we are using the symmetric version of this matrix for numerical stability
        M = np.diag(np.sqrt(nc),0) @ self.Omega @ np.diag(np.sqrt(nc),0)
        u = scipy.linalg.eigvalsh(M)
        idx = u.argsort()[::-1]
        eigenvalues = u[idx]

        snr = eigenvalues[1]**2 / eigenvalues[0]

        return snr

    ######################
    ## PART 3 --- SAMPLING
    ######################


    def generateNetwork(self):
        """
        Network nodes at each leaf of the dendro are equivalent.  For each leaf work out the
        probability of connection with other blocks by working up to the root of the tree.
        """
        Omega = self.Omega
        pvec = self.get_partition_at_level(-1).pvec
        nc = [sum(pvec == i) for i in range(pvec.max() + 1)]
        A = sample_block_model(Omega,nc)
        return A