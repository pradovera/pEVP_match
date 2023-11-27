import numpy as np
from matplotlib import pyplot as plt
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
np.random.seed(42)

# define problem of Section 6.1
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
train_nonpar = lambda L, center, radius: beyn(L, center, radius, lhs, rhs,
                                              25, 1e-10, 1)
tol, interp_kind, patch_width, min_patch_deltap = 1e-2, "linear", None, 5

# train
model, ps_train = train(L, train_nonpar, center, radius, interp_kind,
                        patch_width, p_range, tol,
                        min_patch_deltap = min_patch_deltap)

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
plt.plot(np.real(val_ref[:, 0]), ps, 'ro')
plt.plot(np.real(val_app[:, 0]), ps, 'b:')
plt.plot(np.real(val_ref[:, 1:]), ps, 'ro')
plt.plot(np.real(val_app[:, 1:]), ps, 'b:')
plt.legend(['exact', 'approx'])
plt.xlim(-5, 5), plt.ylim(*p_range)
plt.xlabel("Re(lambda)"), plt.ylabel("p")
plt.subplot(142)
plt.plot(np.imag(val_ref[:, 0]), ps, 'ro')
plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
plt.plot(np.imag(val_ref[:, 1:]), ps, 'ro')
plt.plot(np.imag(val_app[:, 1:]), ps, 'b')
plt.legend(['exact', 'approx'])
plt.xlim(-5, 5), plt.ylim(*p_range)
plt.xlabel("Im(lambda)"), plt.ylabel("p")
plt.subplot(143)
plt.plot([0] * len(ps_train), ps_train, 'bx')
plt.ylim(*p_range)
plt.ylabel("sample p-points")
plt.subplot(144)
plt.semilogx(error, ps)
plt.semilogx([tol] * 2, p_range, 'k:')
plt.ylim(*p_range)
plt.xlabel("lambda error"), plt.ylabel("p")
plt.tight_layout(), plt.show()
