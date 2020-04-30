import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score

"""
The metrics module computes various comparisons and scores of alignment
between different sets of partitions.
"""


def calculate_level_comparison_matrix(pvecs, true_pvecs,
                                      score=adjusted_mutual_info_score):
    """
    Compare the partition at each level of a heirarchy with each level of
    the true hierarchy.
    Default score is the overlap_score.
    Output is a score matrix where rows correspond to the predicted
    hierarchy lvls and columns represent the true hierarchy lvls.
    """
    pred_lvls = len(pvecs)
    true_lvls = len(true_pvecs)
    score_matrix = np.zeros((pred_lvls, true_lvls))
    for i in range(pred_lvls):
        for j in range(true_lvls):
            score_matrix[i, j] = score(pvecs[i].pvec_expanded,
                                       true_pvecs[j].pvec_expanded,
                                       average_method="arithmetic")

    return score_matrix


def calculate_precision_recall(score_matrix):
    """
    Calculates the hierarchy precision and recall from a score matrix.
    """
    pred_lvls, true_lvls = np.shape(score_matrix)
    recall = np.max(score_matrix, 0).sum()/true_lvls
    precision = np.max(score_matrix, 1).sum()/pred_lvls
    return precision, recall


def compare_levels(true_pvec, inf_pvec):
    true_levels = [len(np.unique(pv)) for pv in true_pvec]
    inf_levels = np.array([len(np.unique(pv)) for pv in inf_pvec])

    match = [np.argmin(np.abs(inf_levels-tl)) for tl in true_levels]
    diff = [inf_levels[mi]-tl for mi, tl in zip(match, true_levels)]
    return diff
