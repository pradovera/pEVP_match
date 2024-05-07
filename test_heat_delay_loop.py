import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train
np.random.seed(42)

# define problem of section 6.3
N, l, k, f0, f1, t1, t2 = 5000, np.pi, .02, -.1, .05, 1., 2. # scalar constants appearing in the problem
stiff = sparse.diags([-2 * np.ones(N - 1)] + [np.ones(N - 2)] * 2, [0, -1, 1], format = "csr") * (N / l) ** 2
eye = sparse.eye(N - 1, format = "csr", dtype = complex) # identity
# parametric matrix L(z,p) that defines the pEVP
L_base = lambda z, p: k * stiff + (f0 - z - f1 * np.exp(-t1 * z) - p * np.exp(-t2 * z)) * eye
L = lambda z, p: L_base(z - 1, p) # center at z=0

# define parameter range
p_range = [-.1, .1]

# define parameters for training
l_sketch = 30 # number of sketching directions in Beyn's method
lhs = np.random.randn(l_sketch, N - 1) + 1j * np.random.randn(l_sketch, N - 1) # left sketching matrix
rhs = np.random.randn(N - 1, l_sketch) + 1j * np.random.randn(N - 1, l_sketch) # right sketching matrix
train_nonpar = lambda L, center, radius: beyn(L, center, radius, lhs, rhs, 1000, 1e-10, 5)

# read user input
allowed_tags = ["LINEAR", "SPLINE7"]
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input(("Input example_tag:\n(Allowed values: {})\n").format(allowed_tags))

if example_tag == "LINEAR":
    tols = np.logspace(-1, -6, 11)[: 8] # tolerance for outer adaptive loop
    interp_kind = "linear" # interpolation strategy (piecewise-linear hat functions)
    patch_width = None # minimum width of interpolation patches in case of bifurcations (default)
elif example_tag == "SPLINE7":
    tols = np.logspace(-1, -6, 11) # tolerance for outer adaptive loop
    interp_kind = "spline7" # interpolation strategy (degree-7 splines)
    patch_width = 11 # minimum width of interpolation patches in case of bifurcations

# train
nps = []
for tol in tols:
    print("-----\nTraining with tol =", tol)
    ps_train = train(L, train_nonpar, 0., 1., interp_kind, patch_width, p_range, tol)[1]
    nps += [len(ps_train)]

# plot results
plt.figure()
plt.semilogx(tols, nps, '*-')
plt.xlabel("tol"), plt.ylabel("#p-points")
plt.show()
