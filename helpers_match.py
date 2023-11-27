import numpy as np
from scipy.optimize import linear_sum_assignment

def matchBase(dist):
    """
    This function performs a matching operation based on a distance matrix.

    Parameters:
    dist (numpy array): The distance matrix.

    Returns:
    tuple: The matched indices and the distance matrix.
    """
    
    # linear_sum_assignment does not accept inf entries -> remove inf values!
    inf_0, inf_1 = np.all(np.isinf(dist), axis = 1), np.all(np.isinf(dist), axis = 0)
    notinf_0 = np.where(np.logical_not(inf_0))[0]
    notinf_1 = np.where(np.logical_not(inf_1))[0]
    dist_ = dist[notinf_0][:, notinf_1]

    # remove inf entries and perform matching using scipy routine
    # note: for unbalanced matching, it returns a minimal matching of size min(dist.shape)
    match_0_, match_1_ = linear_sum_assignment(dist_)
    
    # reindex to account for removed entries
    match_0, match_1 = list(notinf_0[match_0_]), list(notinf_1[match_1_])

    # now insert back any inf rows and columns
    match_0 += [i for i in notinf_0 if i not in match_0] + list(np.where(inf_0)[0])
    match_1 += [i for i in notinf_1 if i not in match_1] + list(np.where(inf_1)[0])
    
    # sort first index list for simplicity
    idx_0 = np.argsort(match_0)
    match_0, match_1 = np.array(match_0)[idx_0], np.array(match_1)
    if len(match_0) >= len(match_1):
        match_1 = match_1[idx_0[: len(match_1)]]
    else:
        match_1[: len(match_0)] = match_1[idx_0]
    return (match_0, match_1), dist

def match(values_0, values_1):
    """
    This function matches two sets of values based on their distances.

    Parameters:
    values_0 (numpy array): The first set of values.
    values_1 (numpy array): The second set of values.

    Returns:
    tuple: The matched indices and the distance matrix.
    """
    # assemble distance matrix
    dist = np.empty((len(values_0), len(values_1)))
    # improve robustness when inf are involved
    dist[:] = np.inf
    notinf_0 = np.logical_not(np.isinf(values_0))
    dist[notinf_0] = np.abs(np.reshape(values_0[notinf_0], (-1, 1))
                          - np.reshape(values_1, (1, -1)))
    return matchBase(dist)
