import sys
from time import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.io import loadmat
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
np.random.seed(42)

# define problem of Section 6.4
z0, R, alphas = 3e4, 1.5e4, [0., 108.8774]
data = loadmat("gun_data.mat")
data_mat = {key: csc_matrix(data[key]) for key in ["K", "M", "W1", "W2"]}
N_matrix = data_mat["K"].shape[0]
L_base = lambda z, p: (data_mat["K"] - z * p ** 2 * data_mat["M"]
                     + 1j * p * ((z - alphas[0] ** 2) ** .5 * data_mat["W1"]
                               + (z - alphas[1] ** 2) ** .5 * data_mat["W2"]))
L = lambda z, p: L_base(R * z + z0, p) # center at z=0 and normalize

# define parameter range
p_range = [1., 1.4]

# define parameters for training
l_sketch = 30 # number of sketching directions in Beyn's method
lhs = np.random.randn(l_sketch, N_matrix) + 1j * np.random.randn(l_sketch, N_matrix) # left sketching matrix
rhs = np.random.randn(N_matrix, l_sketch) + 1j * np.random.randn(N_matrix, l_sketch) # right sketching matrix
train_nonpar = lambda L, center, radius: beyn(L, center, radius, lhs, rhs, 200, 1e-10, 5)

tol = 1e-6 / R # tolerance for outer adaptive loop (need to de-normalize it)
interp_kind = "spline7" # interpolation strategy (degree-7 splines)
patch_width = None # minimum width of interpolation patches in case of bifurcations (default)

# train
t0 = time()
model, ps_train = train(L, train_nonpar, 0., 1., interp_kind, patch_width, p_range, tol)
print("approximation training time: {:.5e}".format(time() - t0))

# test
ps = np.linspace(*p_range, 501) # testing grid
ps_coarse = ps[::5] # coarse testing grid
getApprox = lambda p: evaluate(model, ps_train, p, 0., 1., interp_kind, patch_width)
def getExact(p): # reference solution
    return train_nonpar(lambda z: L(z, p), 0., 1.)
val_app, val_ref, error = runTest(ps, 5, getApprox, getExact, 1) # run testing routine
val_app, val_ref = R * val_app + z0, R * val_ref + z0 # shift to original range
error, tol = R * error, R * tol # must rescale to account for z-normalization

# plot approximation and error
plt.figure(figsize = (15, 5))
plt.subplot(141)
plt.plot(np.real(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 0]), ps, 'b:')
plt.plot(np.real(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 1:]), ps, 'b:')
plt.legend(['exact', 'approx'])
plt.xlim(z0 - R, z0 + R), plt.ylim(*p_range)
plt.xlabel("Re(lambda)"), plt.ylabel("p")
plt.subplot(142)
plt.plot(np.imag(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
plt.plot(np.imag(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 1:]), ps, 'b')
plt.legend(['exact', 'approx'])
plt.xlim(0, R), plt.ylim(*p_range)
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
