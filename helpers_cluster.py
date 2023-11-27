import numpy as np
from helpers_match import matchBase

def findLargestFullDiagSubmatrix(A):
    # merge all bifurcation flags in a smart way
    idx = np.arange(len(A))
    zeros = np.where(np.logical_not(A))
    if len(zeros[0]) == 0: return idx # matrix is already full
    # locate first zero
    zero = (zeros[0][0], zeros[1][0])
    # try removing the column where the first zero is
    good_L = np.where(idx != zero[1])[0]
    idx_L = good_L[findLargestFullDiagSubmatrix(A[good_L][:, good_L])]
    if zero[0] != zero[1]:
        # try removing the row where the first zero is
        good_R = np.where(idx != zero[0])[0]
        idx_R = good_R[findLargestFullDiagSubmatrix(A[good_R][:, good_R])]
        if len(idx_R) > len(idx_L): return idx_R
    return idx_L

def findClusters(d, thresh, kind = "largest", inf = 1e10):
    """
    This function finds clusters in a given distance matrix by comparing best match with all possible 2nd-best matches.

    Parameters:
    d (numpy array): The distance matrix.
    thresh (float): The threshold for determining clusters.
    kind (str, optional): The kind of clustering. Allowed values are "smallest", "largest", "sequential". Defaults to "largest".
    inf (float, optional): The value to replace the diagonal entries of the distance matrix. Defaults to 1e10.

    Returns:
    list: The clusters found in the distance matrix.
    """
    if kind not in ["smallest", "largest"]: kind == "sequential"
    S = len(d)
    j_good = np.where(np.diag(d) != 0.)[0]
    d_opt_base = np.sum(np.diag(d))
    d_ = np.eye(S, dtype = bool)
    for j in j_good:
        # replace diag entry with pseudo-inf (matchBase filters any np.inf out!)
        d_j, d[j, j] = d[j, j], inf
        # try matching now
        p_opt, d_opt = matchBase(d)
        d_opt_j = np.sum(np.diag(d_opt[:, p_opt[1]]))
        if d_opt_j / d_opt_base - 1 < thresh:
            d_[j, p_opt[0] != p_opt[1]] = 1
        d[j, j] = d_j
    d_ = np.logical_or(d_, d_.T)
    unused = list(range(S)) # items yet to be assigned
    clusters = [] # assigned clusters
    while len(unused) > 0: # some items have yet to be assigned
        D = d_[unused][:, unused] # get only unassigned entries
        best_cluster = None
        for i in range(len(unused)): 
            if kind == "largest" and best_cluster is not None and len(unused) - i < len(best_cluster):
                break # impossible to obtain a cluster larger than current
            cluster = np.where(D[i])[0] # feasible cluster
            D_ = D[cluster][:, cluster] # get only feasible entries
            idx_cluster = findLargestFullDiagSubmatrix(D_)
            if (best_cluster is None
             # cluster found is larger than best
             or (kind == "largest" and len(idx_cluster) > len(best_cluster))
             # cluster found is smaller than best
             or (kind == "smallest" and len(idx_cluster) < len(best_cluster))):
                best_cluster = cluster[idx_cluster]
                if kind == "sequential": break # accept first cluster found
        clusters += [[unused[j] for j in best_cluster]] # assign new cluster
        for k in best_cluster[::-1]: unused.pop(k) # flag items as used
    return clusters

def mergeClusters(cluster_list, deltap_list, min_patch_deltap):
    """
    This function merges clusters based on minimal cluster width.

    Parameters:
    cluster_list (list): The list of clusters.
    deltap_list (list): The list of delta p values.
    min_patch_deltap (float): The minimum patch delta p.

    Returns:
    list: The merged clusters.
    """
    S = len(cluster_list)
    clusters = []
    for p_j in range(S): # loop over parameters
        cluster_j = [x[:] for x in cluster_list[p_j]]
        # find start of patch
        j_patch_start = p_j
        while (j_patch_start > 0 and 2 * sum(deltap_list[j_patch_start : p_j + 1]) - deltap_list[p_j] < min_patch_deltap): # check width of left half
            j_patch_start -= 1
        # find end of patch
        j_patch_end = p_j + 1
        while (j_patch_end < S and 2 * sum(deltap_list[p_j : j_patch_end]) - deltap_list[p_j] < min_patch_deltap): # check width of right half
            j_patch_end += 1
        for k in range(j_patch_start, j_patch_end): # loop over patch
            if k == p_j: continue # self patch
            for c in cluster_list[k]: # loop over other cluster
                if len(c) <= 1: continue # singleton
                i = c[0]
                i_j = np.where([i in c_j for c_j in cluster_j])[0][0]
                for j in c[1 :]: # force i and j in same cluster in cluster_j
                    j_j = np.where([j in c_j for c_j in cluster_j])[0][0]
                    if i_j != j_j:
                        cluster_j[i_j] += cluster_j[j_j]
                        cluster_j.pop(j_j)
                        if i_j > j_j: i_j -= 1
                        if len(cluster_j) == 1: break
                if len(cluster_j) == 1: break
            if len(cluster_j) == 1: break
        for j in range(len(cluster_j)):
            cluster_j[j] = list(np.sort(cluster_j[j]))
        clusters += [cluster_j]
    return clusters
