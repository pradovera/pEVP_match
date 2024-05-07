from numpy import abs, eye, argmin, argpartition, isinf, any, logical_or, isnan
from scipy.interpolate import interp1d, make_interp_spline

def interp1d_get_local_idx(p, ps, interp_kind):
    """
    This function computes the index for local-ish interpolation.
    
    Parameters:
    p (float): The value of p at which the interpolation is requested.
    ps (numpy array): The array of p-values at which the function is evaluated.
    interp_kind (str): The type of interpolation. Allowed values are "linear", "nearest", "nearest-up", "previous", and "next".

    Returns:
    int: The index for local-ish interpolation.
    """
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
    """
    This function computes the interpolation weights at p, given support points ps. Returns the interpolant function.
    
    Parameters:
    p (float): The value of p at which the interpolation is requested.
    ps (numpy array): The array of p-values at which the function is evaluated.
    interp_kind (str): The type of interpolation. Allowed values are "linear", "nearest", "nearest-up", "previous", "next", "zero", "slinear", "quadratic", "cubic", and "spline*", with the spline order replacing "*".

    Returns:
    function: A function that takes an array x and returns the interpolated value at p.
    """
    if len(ps) == 1 and interp_kind == "linear":
        raise Exception("not enough points")
    if interp_kind in ["linear", "nearest", "nearest-up", "previous", "next"]:
        j = interp1d_get_local_idx(p, ps, interp_kind)
        if len(ps) == 1 or interp_kind != "linear": return lambda x: x[j]
        if ps[j] > p: j -= 1
        j = min(len(ps) - 2, max(0, j)) # clip to range
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
    """
    Computes interpolation weights at p as interp1d_fast(), but allows also infinite values.

    Parameters:
    p (float): The value of p at which the interpolation is requested.
    ps (numpy array): The array of p-values at which the function is evaluated.
    values (numpy array): The array of function values at the p-values.
    interp_kind (str): The type of interpolation. Allowed values are "linear", "nearest", "nearest-up", "previous", "next".

    Returns:
    float: The interpolated value at p.
    """
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
    if j > -1 and not isinf_eff(values[j]):
        # we can try to widen stencil to the left
        for stencil_l in range(j, 0, -1):
            if isinf_eff(values[stencil_l - 1]): break
        else:
            stencil_l = 0
    if j < S - 1 and not isinf_eff(values[j + 1]):
        # we can try to widen stencil to the right
        for stencil_r in range(j + 2, S):
            if isinf_eff(values[stencil_r]): break
        else:
            stencil_r = S
    if stencil_r <= stencil_l: # inf on both sides of j
        return interp1d_inf(p, ps, values, "nearest") # revert to nearest neighbor
    # try to interpolate over stencil
    if ((stencil_l <= j + 1 and stencil_r > j) # stencil contains p or it almost does
     or (j == -1 and stencil_l == 0) # p is left of stencil and range
     or (j == S - 1 and stencil_r == S)): # p is right of stencil and range
        try:
            interp = interp1d_fast(p, ps[stencil_l : stencil_r], interp_kind)
            return interp(values[stencil_l : stencil_r])
        except:
            pass
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
