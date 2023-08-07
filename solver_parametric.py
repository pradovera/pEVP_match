import numpy as np
from helpers_match import match
from helpers_cluster import findClusters, mergeClusters
from helpers_interp1d import interp1d_get_local_idx, interp1d_fast, interp1d_inf

def matchData(data, return_dist):
    # match data, even unbalanced
    if return_dist: dists = []
    for j, dataj in enumerate(data):
        dataj = data[j]
        Nj = len(dataj)
        if j == 0:
            N = Nj
        else: # perform matching
            if Nj < N: # too few new val: add inf values
                dataj = np.pad(dataj, [(0, N - Nj)], constant_values = np.inf)
            elif len(dataj) > N: # too many new val
                # will need to add inf values in old vals
                # for now, we only add them to the last
                data[j - 1] = np.pad(data[j - 1], [(0, Nj - N)], constant_values = np.inf)
                N = Nj
            # now the problem of matching data[j - 1] and dataj is balanced
            p_opt, d_opt = match(data[j - 1], dataj)
            data[j] = dataj[p_opt[1]]
            if return_dist: dists += [d_opt[:, p_opt[1]]]
    for j, dataj in enumerate(data): # add missing entries in unbalanced case
        Nj = len(dataj)
        if Nj < N:
            data[j] = np.pad(data[j], [(0, N - Nj)], constant_values = np.inf)
            if j < len(data) - 1 and return_dist:
                dists[j] = np.pad(dists[j], [(0, N - Nj)] * 2, constant_values = np.inf)
    if return_dist: return np.array(data), dists
    return np.array(data)

def clusterMatchedData(p, data, dists, d_thresh, min_patch_deltap):
    # cluster data after match
    clusters = []
    for d in dists:
        d_inf = np.isinf(np.diag(d))
        d[d_inf, d_inf] = 0.
        clusters += [findClusters(d, d_thresh)]
    # find effective clusters on each patch by merging nearby clusters
    try:
        deltaps = p[1:] - p[:-1]
    except:
        deltaps = np.array([p[j + 1] - p[j] for j in range(len(p) - 1)])
    return mergeClusters(clusters, deltaps, min_patch_deltap)

def train(L, solve_nonpar, args_nonpar, center, radius, interp_kind,
          patch_width, ps_start, tol, max_iter, d_thresh, min_patch_deltap):
    # train approximation model with parameters:
    # L: lambda function defining matrix in eigenproblem
    # solve_nonpar: lambda function defining non-parametric eigensolver
    # args_nonpar: additional arguments of solve_nonpar
    # center: center of contour (disk)
    # radius: radius of contour (disk)
    # interp_kind: string label of p-interpolation type
    # patch_width: width of stencil for interpolation
    # ps_start: initial grid of sample p-points
    # tol: tolerance epsilon for adaptivity
    # max_iter: maximum number of adaptivity iterations
    # d_thresh: tolerance delta for bifurcation
    # min_patch_deltap: width of stencil for implicit bifurcation management
    ps = np.array(ps_start)
    dps = ps[1:] - ps[:-1]
    if any(abs(dps - dps[0]) > 1e-10):
        raise Exception("ps_start must contain equispaced points")
    ps_next = list(.5 * (ps[:-1] + ps[1:]))
    pre_next = list(range(len(ps_next)))
    dps_next = [.25 * dps[0]] * len(ps_next)
    
    # initial sampling
    data = []
    for p in ps:
        Lp = lambda z: L(z, p)
        data += [solve_nonpar(Lp, center, radius, args_nonpar)]
    # train model
    model_out = matchData(data, d_thresh is not None)
    data = list(model_out[0])
    if d_thresh is not None:
        clusters = clusterMatchedData(ps, *model_out, d_thresh,
                                      min_patch_deltap)
        model_out = (model_out[0], clusters)
    
    for _ in range(int(max_iter)):
        # test model
        print("Adaptive match iteration: test at {} point(s)".format(len(ps_next)))
        val_pre, val_ref = [], []
        for p in ps_next:
            print(".", end = "")
            val_pre += [evaluate(model_out, ps, p, center, radius,
                                 interp_kind, patch_width)]
            Lp = lambda z: L(z, p)
            val_ref += [solve_nonpar(Lp, center, radius, args_nonpar)]
        print()
        to_be_refined = []
        for j in range(len(ps_next)):
            # get prediction error
            Nref, Npre = len(val_ref[j]), len(val_pre[j])
            if Nref == 0 and Npre == 0:
                print("\t no eigenvalues at {}".format(ps_next[j]))
            else:
                v_ref, v_pre = val_ref[j], val_pre[j]
                if Nref < Npre:
                    v_ref = np.pad(v_ref, [(0, Npre - Nref)], constant_values = np.inf)
                elif Nref > Npre:
                    v_pre = np.pad(v_pre, [(0, Nref - Npre)], constant_values = np.inf)
                p_opt, d_opt = match(v_ref, v_pre)
                d_opt_diag = d_opt[p_opt[0], p_opt[1]]
                d_opt_diag = d_opt_diag[np.logical_not(np.isinf(d_opt_diag))]
                d_tot = np.max(d_opt_diag) if len(d_opt_diag) else 0.
                print("\t error at {} = {}".format(ps_next[j], d_tot))
            if d_tot > tol: to_be_refined += [j]
        # refine model
        ps_next_new, pre_next_new, dps_next_new = [], [], []
        if len(to_be_refined) == 0: break
        ps_new = list(ps)
        for n_added, j in enumerate(to_be_refined):
            ctr, pre, step = ps_next[j], pre_next[j], dps_next[j]
            for ishift, shift in enumerate([- step, step]):
                ctrn = ctr + shift
                if len(ps_next_new) == 0 or abs(ctrn - ps_next_new[-1]) > 1e-10: # no duplicate entries
                    ps_next_new += [ctrn]
                    pre_next_new += [pre + n_added + ishift]
                    dps_next_new += [.5 * step]
            idx_add = pre + n_added + 1
            ps_new = ps_new[: idx_add] + [ctr] + ps_new[idx_add :]
            data = data[: idx_add] + [val_ref[j]] + data[idx_add :]
        ps, ps_next = np.array(ps_new), ps_next_new
        pre_next, dps_next = pre_next_new, dps_next_new
        # train model
        model_out = matchData(data, d_thresh is not None)
        data = list(model_out[0])
        if d_thresh is not None:
            clusters = clusterMatchedData(ps, *model_out, d_thresh,
                                          min_patch_deltap)
            model_out = (model_out[0], clusters)
    else:
        print("Max number of refinement iterations reached!")
    return model_out, ps

def evaluate(model, ps, p, center, radius, interp_kind, patch_width):
    # evaluate approximation model with parameters:
    # model: first output of "train" above
    # ps: second output of "train" above
    # p: value of p at which the evaluation is requested
    # center: center of contour (disk)
    # radius: radius of contour (disk)
    # interp_kind: string label of p-interpolation type
    # patch_width: width of stencil for interpolation
    S = len(ps)
    has_clusters = isinstance(model, tuple)
    j = interp1d_get_local_idx(p, ps, "previous")
    if j >= S - 1: j = S - 2
    j_patch_start, j_patch_end = 0, S
    if patch_width is not None: # width is patch_width + 1
        j_patch_start = max(0, j - (patch_width - 1) // 2)
        j_patch_end = min(S, j + (patch_width + 3) // 2)
    S_eff = j_patch_end - j_patch_start
    ps_eff = ps[j_patch_start : j_patch_end]
    interp = interp1d_fast(p, ps_eff, interp_kind)
    if has_clusters:
        data, cluster = model[0], model[1][j]
    else:
        data, cluster = model, [[j] for j in range(model.shape[1])]
    N = data.shape[1]
    values = np.empty(N, dtype = complex)

    for c in cluster:
        if len(c) == 1: # explicit form
            c_ = c[0]
            if np.any(np.isinf(data[j_patch_start : j_patch_end, c_])):
                values[c_] = interp1d_inf(p, ps_eff, data[j_patch_start : j_patch_end, c_], interp_kind)
            else:
                values[c_] = interp(data[j_patch_start : j_patch_end, c_])
        else: # implicit form
            poly = np.empty((S_eff, len(c) + 1), dtype = complex)
            for k in range(j_patch_start, j_patch_end):
                poly[k - j_patch_start] = np.poly(data[k, c])
            if np.any(np.isinf(data[j_patch_start : j_patch_end, c])):
                poly_interpolated = interp1d_inf(p, ps_eff, poly, interp_kind)
                poly_interpolated[0] = 1.
            else:
                poly_interpolated = interp(poly)
            try:
                values[c] = np.roots(poly_interpolated)
            except np.linalg.LinAlgError:
                values[c] = np.inf
    return values[np.abs(values - center) <= radius]
