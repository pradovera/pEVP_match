import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train
np.random.seed(42)

# read user input
allowed_tags = ["LINEAR", "SPLINE7"]
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input(("Input example_tag:\n(Allowed values: {})\n").format(allowed_tags))

# define problem of section 6.3
z_shift = -1.
N, l, k, f0, f1, t1, t2 = 5000, np.pi, .02, -.1, .05, 1., 2.
stiff = sparse.diags([-2 * np.ones(N - 1)] + [np.ones(N - 2)] * 2, [0, -1, 1], format = "csr") * (N / l) ** 2
eye = sparse.eye(N - 1, format = "csr", dtype = complex)
L_base = lambda z, p: k * stiff + (f0 - z - f1 * np.exp(-t1 * z) - p * np.exp(-t2 * z)) * eye
L = lambda z, p: L_base(z + z_shift, p) # center at z=0
p_range = [-.1, .1]

# define parameters for training
l_sketch = 30
lhs = np.random.randn(l_sketch, N - 1) + 1j * np.random.randn(l_sketch, N - 1)
rhs = np.random.randn(N - 1, l_sketch) + 1j * np.random.randn(N - 1, l_sketch)
train_nonpar = lambda L, center, radius: beyn(L, center, radius, lhs, rhs,
                                              1000, 1e-10, 5)
if example_tag == "LINEAR":
    tols, interp_kind, patch_width = np.logspace(-1, -6, 11)[: 8], "linear", None
elif example_tag == "SPLINE7":
    tols, interp_kind, patch_width = np.logspace(-1, -6, 11), "spline7", 11

# train
nps = []
for tol in tols:
    print("-----\nTraining with tol =", tol)
    ps_train = train(L, train_nonpar, 0., 1., interp_kind,
                     patch_width, p_range, tol)[1]
    nps += [len(ps_train)]

# plot results
plt.figure()
plt.semilogx(tols, nps, '*-')
plt.xlabel("tol"), plt.ylabel("#p-points")
plt.show()
