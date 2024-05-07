import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
np.random.seed(42)

# define problem of Section 6.3
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
allowed_tags = ["NORMAL", "FINE"]
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input(("Input example_tag:\n(Allowed values: {})\n").format(allowed_tags))
if example_tag == "NORMAL":
    tol = 1e-2 # tolerance for outer adaptive loop
    interp_kind = "spline3" # interpolation strategy (cubic splines)
    patch_width = 7 # minimum width of interpolation patches in case of bifurcations
elif example_tag == "FINE":
    tol = 1e-6 # tolerance for outer adaptive loop
    interp_kind = "spline7" # interpolation strategy (degree-7 splines)
    patch_width = 11 # minimum width of interpolation patches in case of bifurcations

# train
model, ps_train = train(L, train_nonpar, 0., 1., interp_kind, patch_width, p_range, tol)

# test
ps = np.linspace(*p_range, 500) # testing grid
ps_coarse = ps[::10] # coarse testing grid
getApprox = lambda p: evaluate(model, ps_train, p, 0., 1., interp_kind, patch_width)
def getExact(p): # reference solution
    return train_nonpar(lambda z: L(z, p), 0., 1.)
val_app, val_ref, error = runTest(ps, 10, getApprox, getExact) # run testing routine
val_app, val_ref = val_app - 1, val_ref - 1 # shift to original range

# plot approximation and error
plt.figure(figsize = (15, 5))
plt.subplot(141)
plt.plot(np.real(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 0]), ps, 'b:')
plt.plot(np.real(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 1:]), ps, 'b:')
plt.legend(['exact', 'approx'])
plt.xlim(-2, 0), plt.ylim(*p_range)
plt.xlabel("Re(lambda)"), plt.ylabel("p")
plt.subplot(142)
plt.plot(np.imag(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
plt.plot(np.imag(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 1:]), ps, 'b')
plt.legend(['exact', 'approx'])
plt.xlim(-1, 1), plt.ylim(*p_range)
plt.xlabel("Im(lambda)"), plt.ylabel("p")
plt.subplot(143)
plt.plot([0] * len(ps_train), ps_train, 'bx')
plt.ylim(*p_range)
plt.ylabel("sample p-points")
plt.subplot(144)
plt.semilogx(error, ps_coarse)
plt.semilogx([tol] * 2, p_range, 'k:')
plt.ylim(*p_range)
plt.xlabel("lambda error"), plt.ylabel("p")
plt.tight_layout(), plt.show()
