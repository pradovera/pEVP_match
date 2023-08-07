import numpy as np
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
np.random.seed(42)

# define problem
A = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 0]])
B = np.array([[0, 0, -2], [0, 0, -1], [0, 0, 0]])
C = np.eye(3)
L = lambda z, p: A + p * B - z * C
p_range = [-50., 50.]

# define parameters for training
center, radius = 0., 4.
l_sketch = 5
lhs = np.random.randn(l_sketch, 3) + 1j * np.random.randn(l_sketch, 3)
rhs = np.random.randn(3, l_sketch) + 1j * np.random.randn(3, l_sketch)
train_nonpar = lambda L, center, radius, args_nonpar: beyn(L, center, radius, *args_nonpar)
args_nonpar = (lhs, rhs, 25, 1e-10)
ps_train, tol, interp_kind = p_range, 1e-2, "linear"
patch_width, d_thresh, min_patch_deltap = None, 1e-1, 5

# train
model, ps_train = train(L, train_nonpar, args_nonpar, center, radius, interp_kind,
                        patch_width, ps_train, tol, 100, d_thresh, min_patch_deltap)

# test
ps = np.linspace(*p_range, 1500)
getApprox = lambda p: evaluate(model, ps_train, p, center, radius, interp_kind, patch_width)
def getExact(p): # exact solution
    alpha = lambda p: (((3*(4*p**3+84*p**2-60*p-5+0j))**.5-18*p+9)/18) ** (1./3)
    beta = lambda p: (p-2)/3/alpha(p)
    versor1, versor2 = np.exp(1j*np.pi/3), np.exp(-1j*np.pi/3)
    val1 = lambda p: alpha(p)-beta(p)
    val2 = lambda p: -versor2*alpha(p)+versor1*beta(p)
    val3 = lambda p: -versor1*alpha(p)+versor2*beta(p)
    v_ref = np.array([val1(p), val2(p), val3(p)])
    return v_ref[np.abs(v_ref - center) <= radius]
val_app, val_ref, error = runTest(ps, 1, getApprox, getExact)

# plot approximation and error
plt.figure(figsize = (15, 5))
plt.subplot(141)
plt.plot(np.real(val_ref), ps, 'r')
plt.plot(np.real(val_app), ps, 'b:')
plt.xlim(-5, 5), plt.ylim(*p_range)
plt.subplot(142)
plt.plot(np.imag(val_ref), ps, 'r')
plt.plot(np.imag(val_app), ps, 'b:')
plt.xlim(-5, 5), plt.ylim(*p_range)
plt.subplot(143)
plt.plot([0] * len(ps_train), ps_train, 'bx')
plt.ylim(*p_range)
plt.subplot(144)
plt.semilogx(error, ps)
plt.semilogx([tol] * 2, p_range, 'k:')
plt.ylim(*p_range)
plt.tight_layout(), plt.show()
