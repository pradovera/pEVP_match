import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
np.random.seed(42)

# read user input
allowed_tags = ["NORMAL", "FINE"]
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input(("Input example_tag:\n(Allowed values: {})\n").format(allowed_tags))

# define problem
N, l, k, f0, f1, t1, t2 = 5000, np.pi, .02, -.1, .05, 1., 2.
stiff = sparse.diags([-2 * np.ones(N - 1)] + [np.ones(N - 2)] * 2, [0, -1, 1], format = "csr") * (N / l) ** 2
eye = sparse.eye(N - 1, format = "csr", dtype = complex)
L = lambda z, p: k * stiff + (f0 - (z - 1.) - f1 * np.exp(-t1 * (z - 1.)) - p * np.exp(-t2 * (z - 1.))) * eye
p_range = [-.1, .1]

# define parameters for training
center, radius = 0., 1.
l_sketch = 30
lhs = np.random.randn(l_sketch, N - 1) + 1j * np.random.randn(l_sketch, N - 1)
rhs = np.random.randn(N - 1, l_sketch) + 1j * np.random.randn(N - 1, l_sketch)
train_nonpar = lambda L, center, radius, args_nonpar: beyn(L, center, radius, *args_nonpar)
args_nonpar = (lhs, rhs, 1000, 1e-10, 5)
d_thresh, min_patch_deltap = 1e-1, 1e-2
if example_tag == "NORMAL":
    tol, interp_kind, patch_width = 1e-2, "spline3", 7
elif example_tag == "FINE":
    tol, interp_kind, patch_width = 1e-6, "spline7", 11

# train
model, ps_train = train(L, train_nonpar, args_nonpar, center, radius, interp_kind,
                        patch_width, p_range, tol, 100, d_thresh, min_patch_deltap)

# test
ps = np.linspace(*p_range, 500)
ps_coarse = ps[::10]
getApprox = lambda p: evaluate(model, ps_train, p, center, radius, interp_kind, patch_width)
def getExact(p): # reference solution
    v_ref = train_nonpar(lambda z: L(z, p), center, radius, args_nonpar)
    return v_ref[np.abs(v_ref - center) <= radius]
val_app, val_ref, error = runTest(ps, 10, getApprox, getExact)

# plot approximation and error
plt.figure(figsize = (15, 5))
plt.subplot(141)
plt.plot(np.real(val_ref), ps_coarse, 'ro')
plt.plot(np.real(val_app), ps, 'b:')
plt.xlim(-1, 1), plt.ylim(*p_range)
plt.subplot(142)
plt.plot(np.imag(val_ref), ps_coarse, 'ro')
plt.plot(np.imag(val_app), ps, 'b:')
plt.xlim(-1, 1), plt.ylim(*p_range)
plt.subplot(143)
plt.plot([0] * len(ps_train), ps_train, 'bx')
plt.ylim(*p_range)
plt.subplot(144)
plt.semilogx(error, ps_coarse)
plt.semilogx([tol] * 2, p_range, 'k:')
plt.ylim(*p_range)
plt.tight_layout(), plt.show()
