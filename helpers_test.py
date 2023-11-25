import numpy as np
from helpers_match import match

def runTest(ps, coarsen, get_approx, get_exact):
    ps_coarse = ps[::coarsen]
    val_app = np.empty((len(ps), 0), dtype = complex)
    val_ref = np.empty((len(ps_coarse), 0), dtype = complex)
    error = np.empty((len(ps_coarse), 0))
    val_ref[:], val_app[:], error[:] = np.inf, np.inf, np.nan
    for j, p in enumerate(ps):
        v_app = get_approx(p)
        Napp = len(v_app)
        if Napp > val_app.shape[1]: # enlarge arrays for storage
            dN = Napp - val_app.shape[1]
            val_app = np.pad(val_app, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
        val_app[j, : Napp] = v_app
        
        # compute error
        if not j % coarsen:
            v_ref = get_exact(p)
            Nref = len(v_ref)
            if Nref > val_ref.shape[1]: # enlarge arrays for storage
                dN = Nref - val_ref.shape[1]
                val_ref = np.pad(val_ref, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
                error = np.pad(error, [(0, 0), (0, dN)], constant_values = np.nan)
            val_ref[j // coarsen, : Nref] = v_ref

            if Nref == Napp:
                p_opt, d_opt = match(v_ref, v_app)
            elif Nref < Napp:
                p_opt, d_opt = match(np.pad(v_ref, [(0, Napp - Nref)], constant_values = np.inf), v_app)
            else: #if Nref > Napp:
                p_opt, d_opt = match(v_ref, np.pad(v_app, [(0, Nref - Napp)], constant_values = np.inf))
            error[j // coarsen, p_opt[0]] = d_opt[p_opt[0], p_opt[1]]
    return val_app, val_ref, error