import numpy as np
from helpers_match import match

def runTest(ps, coarsen, get_approx, get_exact):
    """
    This function runs a test to compare the approximation of a function with its exact value.

    Parameters:
    ps (numpy array): The array of p-values at which the function is evaluated.
    coarsen (int): The coarsening factor for the p-values.
    get_approx (function): The function that computes the approximation.
    get_exact (function): The function that computes the exact value.

    Returns:
    val_app (numpy array): The approximated values.
    val_ref (numpy array): The exact values.
    error (numpy array): The error between the approximated and exact values.
    """
    ps_coarse = ps[::coarsen]
    val_app = np.empty((len(ps), 0), dtype = complex)
    val_ref = np.empty((len(ps_coarse), 0), dtype = complex)
    error = np.empty((len(ps_coarse), 0))
    val_ref[:], val_app[:], error[:] = np.inf, np.inf, np.nan
    for j, p in enumerate(ps):
        v_app = get_approx(p)
        Napp = len(v_app)
        dN = Napp - val_app.shape[1]
        if dN > 0: # enlarge arrays for storage
            val_app = np.pad(val_app, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
        elif dN < 0:
            v_app = np.pad(v_app, (0, - dN), constant_values = np.inf + 1j*np.inf)
            Napp = val_app.shape[1]
        # sort v to follow trajectories (note: here val_app.shape[1] == len(v_app) so the match problem is square)
        if j > 0:
            p_opt, _ = match(val_app[j - 1, :], v_app)
            v_app = v_app[p_opt[1]]
        val_app[j, :] = v_app
        
        # compute error
        if not j % coarsen:
            v_ref = get_exact(p)
            Nref = len(v_ref)
            dN = Nref - val_ref.shape[1]
            if dN > 0: # enlarge arrays for storage
                val_ref = np.pad(val_ref, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
                error = np.pad(error, [(0, 0), (0, dN)], constant_values = np.nan)
            elif dN < 0:
                v_ref = np.pad(v_ref, (0, - dN), constant_values = np.inf + 1j*np.inf)
                Nref = val_ref.shape[1]
            # sort v to follow trajectories (note: here val_ref.shape[1] == len(v_ref) so the match problem is square)
            if j > 0:
                p_opt, _ = match(val_ref[j // coarsen - 1, :], v_ref)
                v_ref = v_ref[p_opt[1]]
            val_ref[j // coarsen, :] = v_ref

            if Nref == Napp:
                p_opt, d_opt = match(v_ref, v_app)
            elif Nref < Napp:
                p_opt, d_opt = match(np.pad(v_ref, [(0, Napp - Nref)], constant_values = np.inf), v_app)
                p_opt, d_opt = (p_opt[0][: Nref], p_opt[1][: Nref]), d_opt[: Nref] # remove extras
            else: #if Nref > Napp:
                p_opt, d_opt = match(v_ref, np.pad(v_app, [(0, Nref - Napp)], constant_values = np.inf))
            error[j // coarsen, p_opt[0]] = d_opt[p_opt[0], p_opt[1]]
    return val_app, val_ref, error
