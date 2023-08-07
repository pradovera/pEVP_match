from numpy import abs, eye, argmin, argpartition, isinf, any, logical_or, isnan
from scipy.interpolate import interp1d, make_interp_spline

def interp1d_get_local_idx(p, ps, interp_kind):
    # get index for local-ish interpolation
    # allowed values of interp_kind: linear, nearest, nearest-up, previous, next
    if interp_kind not in ["linear", "nearest", "nearest-up", "previous", "next"]:
        raise Exception("Value of interp_kind not allowed")
    if p <= ps[0]: return 0
    if p >= ps[-1]: return len(ps) - 1
    if interp_kind in ["nearest", "nearest-up"]:
        j, jp = argpartition(abs(ps - p), 1)[: 2]
        if abs(ps[j] - p) == abs(ps[jp] - p):
            if interp_kind == "nearest-up":
                return min(j, jp)
            return max(j, jp)
        return j
    # if interp_kind in ["previous", "next"]:
    j = argmin(abs(ps - p))
    if (interp_kind == "linear" or ps[j] == p
     or (interp_kind == "previous" and ps[j] < p)
     or (interp_kind == "next" and ps[j] > p)): return j
    if interp_kind == "previous": return j - 1
    return j + 1

def interp1d_fast(p, ps, interp_kind):
    # interpolation weights at p, given support points ps
    # allowed values of interp_kind: linear, nearest, nearest-up, previous, next, zero, slinear, quadratic, cubic
    # also allowed is "spline*", with the spline order replacing "*"
    if interp_kind in ["linear", "nearest", "nearest-up", "previous", "next"]:
        if p <= ps[0]: return lambda x: x[0] # clip to range
        if p >= ps[-1]: return lambda x: x[-1] # clip to range
        j = interp1d_get_local_idx(p, ps, interp_kind)
        if interp_kind != "linear": return lambda x: x[j]
        if ps[j] > p: j -= 1
        wp = (ps[j + 1] - p) / (ps[j + 1] - ps[j])
        w = 1. - wp
        return lambda x: wp * x[j] + w * x[j + 1]
    N = len(ps) # number of parameter samples
    if interp_kind in ["zero", "slinear", "quadratic", "cubic"]:
        try:
            weights = interp1d(ps, eye(N), interp_kind, fill_value = "extrapolate")(p)
        except ValueError:
            if interp_kind == "slinear":
                interp_kind = "zero"
            elif interp_kind == "quadratic":
                interp_kind = "slinear"
            elif interp_kind == "cubic":
                interp_kind = "quadratic"
            return interp1d_fast(p, ps, interp_kind)
    elif interp_kind[:6] == "spline":
        k_spline = int(interp_kind[6:])
        try:
            weights = make_interp_spline(ps, eye(N), k_spline)(p)
        except ValueError:
            return interp1d_fast(p, ps, interp_kind[:6] + str(k_spline - 1))
    else:
        raise Exception("Value of interp_kind not recognized")
    return lambda x: sum([w * x_ for w, x_ in zip(weights, x)])

def interp1d_inf(p, ps, values, interp_kind):
    # interpolation weights at p, allowing also infinite values
    if hasattr(values[0], "__len__") and len(values[0]) > 1:
        isinf_eff = lambda x: any(logical_or(isinf(x), isnan(x)))
    else:
        isinf_eff = isinf
    S = len(ps)
    if interp_kind in ["nearest", "nearest-up", "previous", "next"]:
        # since we just take a nearby value, no real interpolation is
        #   needed, so the stencil can just be the whole interval
        j = interp1d_get_local_idx(p, ps, interp_kind)
        return values[j]
    # check if evaluation is at training point
    j = interp1d_get_local_idx(p, ps, "nearest")
    if abs(ps[j] - p) < 1e-12: return values[j]
    # find local stencil where value is finite
    if ps[j] > p: j -= 1
    stencil_l, stencil_r = max(j + 1, 0), min(j + 1, S)
    if j >= 0 and not isinf_eff(values[j]):
        # we can try to widen stencil to the left
        stencil_l = j - 1
        for l in range(j - 1, -1, -1):
            if isinf_eff(values[l]): break
        stencil_l += 1
    if j + 1 < S and not isinf_eff(values[j + 1]):
        # we can try to widen stencil to the right
        stencil_r = j + 2
        for stencil_r in range(j + 2, S):
            if isinf_eff(values[stencil_r]): break
        else:
            stencil_r = S
    # try to interpolate over stencil
    if (j < 0 or stencil_l <= j) and (j + 1 >= S or stencil_r > j + 1):
        # stencil contains p!
        try:
            interp = interp1d_fast(p, ps[stencil_l : stencil_r], interp_kind)
            return interp(values[stencil_l : stencil_r])
        except:
            pass
    # stencil is too small for correct interpolation at p
    if (j < 0 or j + 1 >= S # outside range
     or stencil_r < stencil_l + 1): # inf on both sides of j
        # revert to nearest neighbor
        return interp1d_inf(p, ps, values, "nearest")
    interp = interp1d_fast(p, ps[j : j + 2], "linear")
    if stencil_l <= j and stencil_r > j + 1: # no inf on either side of j
        # revert to linear interpolant on interval
        return interp(values[stencil_l : stencil_r])
    # resort to reciprocal of linear interpolant on interval
    if stencil_l <= j: # inf on right side of j
        weight_l = interp([1, 0])
        return values[j] / weight_l
    weight_r = interp([0, 1])
    return values[j + 1] / weight_r
