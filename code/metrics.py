from __future__ import division
import numpy as np
from scipy import sparse
import scipy
from scipy.optimize import linear_sum_assignment

#calculate a distance matrix based on variation of information
def calcVI(partitions):

    num_partitions,n=np.shape(partitions)
    nodes = np.arange(n)
    c=len(np.unique(partitions))
    vi_mat=np.zeros((num_partitions,num_partitions))


    for i in xrange(num_partitions):
        A1 = sparse.coo_matrix((np.ones(n),(partitions[i,:],nodes)),shape=(c,n),dtype=np.uint).tocsc()
        n1all = np.array(A1.sum(1),dtype=float)

        for j in xrange(i):

            A2 = sparse.coo_matrix((np.ones(n),(nodes,partitions[j,:])),shape=(n,c),dtype=np.uint).tocsc()
            n2all = np.array(A2.sum(0),dtype=float)

            n12all = np.array(A1.dot(A2).todense(),dtype=float)

            rows, columns = n12all.nonzero()

            vi = np.sum(n12all[rows,columns]*np.log((n12all*n12all/(np.outer(n1all,n2all)))[rows,columns]))

            vi = -1/(n*np.log(n))*vi
            vi_mat[i,j]=vi
            vi_mat[j,i]=vi

    return vi_mat


def fraction_correctly_aligned(partition1,partition2):
    """
    Compares two partitions and computes the fraction of nodes in partition1 that can be
    aligned with the nodes in partition2
    """

    pmatrix1 = create_partition_matrix_from_vector(partition1)
    pmatrix2 = create_partition_matrix_from_vector(partition2)

    # cost is minimized --- overlap maximized
    cost_matrix = -pmatrix1.T.dot(pmatrix2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.A)
    cost = -cost_matrix[row_ind, col_ind].sum()

    return cost


def overlap_score(partition, true_partition):
    """
    Compare the overlap score as defined, e.g., in Krzakala et al's spectral redemption paper
    """
    raw_overlap = fraction_correctly_aligned(partition, true_partition)
    num_nodes = partition.size
    num_groups = true_partition.max() +1
    num_groups2 = partition.max() +1
    # print raw_overlap/num_nodes

    # TODO: this might not be necessary!? See below
    if num_groups2 != num_groups:
        print "partitions with different number of groups"

    if num_groups < num_groups2:
        num_groups = num_groups2

    overlap_score  = (raw_overlap/num_nodes - 1/num_groups)/(1-1/num_groups)
    return overlap_score


def calculate_level_comparison_matrix(pvecs,true_pvecs,score=overlap_score):
    """
    Compare the partition at each level of a heirarchy with each level of the true hierarchy.
    Default score is the overlap_score.
    Output is a score matrix where rows correspond to the predicted hierarchy lvls and
    columns represent the true hierarchy lvls.
    """
    pred_lvls=len(pvecs)
    true_lvls=len(true_pvecs)
    score_matrix=np.zeros((pred_lvls,true_lvls))
    for i in xrange(pred_lvls):
        for j in xrange(true_lvls):
            score_matrix[i,j]=score(pvecs[i],true_pvecs[j])

    return score_matrix


def calculate_precision_recall(score_matrix):
    """
    Calculates the hierarchy precision and recall from a score matrix
    """
    pred_lvls,true_lvls = np.shape(score_matrix)
    recall = np.max(score_matrix,0).sum()/true_lvls
    precision = np.max(score_matrix,1).sum()/pred_lvls
    return precision, recall


def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    #~ k=len(np.unique(partition_vec))
    k=np.max(partition_vec)+1 # changed to max value in case some groups are empty

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix
