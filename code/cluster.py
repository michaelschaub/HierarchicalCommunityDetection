#!/usr/bin/env python
"""
Clustering methods
and partition processing methods
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import mixture
from scipy import sparse
from scipy import linalg
import error_handling


def find_partition(evecs, k, method='KM', n_init=20, normalization=True):
    """Perform different types of clustering in spectral embedding space."""
    V = evecs[:, :k]

    if normalization:
        X = preprocessing.normalize(V, axis=1, norm='l2')
    else:
        X = V

    # select methof of clustering - QR, KM (k-means), or GMM
    if method == 'QR':
        partition_vec = clusterEVwithQR(X)
    elif method == 'KM':
        clust = KMeans(n_clusters=k, n_init=n_init, n_jobs=2)
    elif method == 'GMM':
        clust = mixture.GaussianMixture(n_components=k, n_init=n_init,
                                        covariance_type='full')
    elif method == 'GMMspherical':
        clust = mixture.GaussianMixture(n_components=k, n_init=n_init,
                                        covariance_type='spherical')
    elif method == 'GMMdiag':
        clust = mixture.GaussianMixture(n_components=k, n_init=n_init,
                                        covariance_type='diag')
    else:
        raise ValueError('''something went wrong. Please specify a valid
                            clustering method''')

    clust.fit(X)
    partition_vec = clust.predict(X)

    part = Partition(pvec=partition_vec)

    return part


def clusterEVwithQR(evecs, randomized=False, gamma=4):
    """Given a set of eigenvectors find the clusters of the SBM."""
    raise NotImplementedError('Not currently in use')
    if randomized is True:
        Z, Q = orthogonalizeQR_randomized(evecs, gamma)
    else:
        Z, Q = orthogonalizeQR(evecs)

    cluster_ = scipy.absolute(Z).argmax(axis=1).astype(int)

    return cluster_


def project_orthogonal_to(subspace_basis, vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or
    normalized)
    vectors that span the space orthogonal to which we want to project
    vectors_to_project: project these vectors into the orthogonal
    complement of the specified subspace

    compute S*(S^T*S)^{-1}*S' * V
    """
    V = vectors_to_project

    S = subspace_basis

    projected = S @ sparse.linalg.spsolve(S.T @ S, S.T @ V)

    orthogonal_proj = V - projected
    return orthogonal_proj


def add_noise_to_small_matrix(M, snr=0.001, noise_type="gaussian"):
    """Add some small random noise to a (dense) small
    matrix as a perturbation"""
    # noise level is taken relative to the Froebenius norm
    normM = linalg.norm(M, 2)

    if noise_type == "uniform":
        # TODO -- should we have uniform noise?
        NotImplementedError('Noise type uniform not implemented')
    elif noise_type == "gaussian":
        n, m = M.shape
        noise = np.triu(np.random.randn(n, m))
        noise = noise + noise.T - np.diag(np.diag(noise))
        normNoise = linalg.norm(noise, 2)
        Mp = M + snr * normM / normNoise * noise

    return Mp


class Hierarchy(list):

    # Question: it seems useful to have a constructor with a list of partitions/or an empty Hierarchy (no partition)?
    def __init__(self, partition):
        assert type(partition) == Partition
        self.append(partition)
        self.n = partition.pvec.size
        self.k = len(partition)

    def add_level(self, partition):
        assert type(partition) == Partition
        if partition.pvec.size == len(self[-1]):
            self.append(partition)
        else:
            raise NotImplementedError("partition should be a refinement of the previous partition")

    def expand_partitions_to_full_graph(self):
        """
        Map list of aggregated partition vectors in hierarchy to list of
        full-sized partition vectors
        """
        # the finest partition is already at the required size
        self[0].pvec_expanded = self[0].pvec

        # loop over all other partition
        for p0, partition in zip(self[:-1], self[1:]):
            # group indices of previous level correspond to nodes in the
            # aggregated graph;
            # get the group ids of those nodes, and expand by reading out one
            # index per previous node
            partition.pvec_expanded = partition.pvec[p0.pvec_expanded]


class Partition(object):
    """
    Partition class.

    Attributes:
    pvec   : vector of group assignments
    H       : partition matrix (n x k)
    Hnorm   : normalized partition matrix
    proj_error: projection error

    Functions:
    calculate_proj_error
    relabel_partition_vec
    create_partition_matrix
    count_links_between_groups
    """

    def __init__(self, pvec):
        self.pvec = pvec
        self.pvec_expanded = pvec
        self.relabel_partition_vec()
        self.k = len(np.unique(pvec))
        # below sets self.H and self.Hnorm
        self.create_partition_matrix()

    def __len__(self):
        return self.k

    def calculate_proj_error(self, evecs, norm="Fnew", useHnorm=True):
        """ Given a set of eigenvectors and a partition matrix,
        try project compute the alignment between those two subpacees
        by computing the projection (errors) of one into the other"""
        if useHnorm:
            H = self.Hnorm
        else:
            raise NotImplementedError("""This functionality is implemented!
                                         Remove this error""")
            H = self.H
        n, k = np.shape(H)
        if n == k:
            error = 0
            return error
        V = evecs[:, :k]
        proj1 = project_orthogonal_to(H, V)

        if norm == 'Fnew':
            norm1 = linalg.norm(proj1)
            error = norm1**2
        else:
            raise NotImplementedError("Not defined")

        self.proj_error = error
        return error

    def relabel_partition_vec(self):
        """
        Relabel partition vector.

        Given a partition vectors pvec, relabel the groups such that the new
        partition vector has contiguous group labels (starting with 0)
        """
        pvec = self.pvec
        old_labels = np.unique(pvec)
        remap = dict((k, v) for v, k in enumerate(old_labels))

        remap_generator = (remap[i] for i in pvec)
        self.pvec = np.fromiter(remap_generator, int, len(pvec))

    def create_partition_matrix(self):
        """Create a partition indicator matrix from a given vectors."""
        pvec = self.pvec
        nr_nodes = pvec.size

        H = sparse.coo_matrix((np.ones(nr_nodes), (np.arange(nr_nodes), pvec)),
                       shape=(nr_nodes, self.k)).tocsr()
        self.H = H
        self.Hnorm = preprocessing.normalize(H, axis=0, norm='l2')

    def count_links_between_groups(self, A, directed=True, self_loops=False):
        """
        Compute the number of possible and actual links between the groups
        indicated in the partition vector.
        """

        H = self.H

        # each block counts the number of half links / directed links
        links_between_groups = (H.T @ A @ H).A

        # convert to array type first, before performing outer product
        nodes_per_group = np.ravel(H.sum(0))
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
