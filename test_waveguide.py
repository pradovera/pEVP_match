import numpy as np
from matplotlib import pyplot as plt
import fenics as fen
import mshr
from solver_nonparametric import beyn
from solver_parametric import train, evaluate
from helpers_test import runTest
from helpers_fenics import fenics2Sparse
np.random.seed(42)

# define problem of Section 6.2
center, radius = 5., 2.
class engine:
    def L(self, z, p):
        # if the matrices have already been assembled for the current p,
        # we can skip expensive meshing and assembly
        # 'pLastAssembled' keeps track of this
        if not hasattr(self, "pLastAssembled") or self.pLastAssembled != p:
            # perform new meshing and FEM assembly
            domain = (mshr.Rectangle(fen.Point(0., 0.), fen.Point(3., 1.))
                    - mshr.Circle(fen.Point(1., 0.), p))
            print("Meshing for p =", p)
            mesh = mshr.generate_mesh(domain, 50)
            Vspace = fen.FunctionSpace(mesh, "P", 2)

            RobinBoundary = fen.AutoSubDomain(lambda x, on_boundary: (
                                            on_boundary and x[0] > 3. - 1e-10))
            robinMarker = fen.MeshFunction("size_t", mesh, 1)
            RobinBoundary.mark(robinMarker, 1)
            ds = fen.Measure("ds", domain = mesh, subdomain_data = robinMarker)
            
            u, v = fen.TrialFunction(Vspace), fen.TestFunction(Vspace)
            stiff = fen.inner(fen.grad(u), fen.grad(v)) * fen.dx
            outlet = u * v * ds(1)
            mass = u * v * fen.dx
            self.Ls = [fenics2Sparse(stiff),
                     - 1j * fenics2Sparse(outlet),
                     - fenics2Sparse(mass)]
            self.pLastAssembled = p
        return self.Ls[0] + z * self.Ls[1] + z ** 2 * self.Ls[2]

engine_fenics = engine()
L = lambda z, p: engine_fenics.L(z * radius + center, p) # center at z=0
p_range = [.2, .8]

# define parameters for training
l_sketch = 10
def lhs_rhs(L):
    # generate random matrices for left- and right-sketching
    # since the problem size varies with p, we need to generate new ones as p changes
    N = L(0.).shape[0]
    return (np.random.randn(l_sketch, N) + 1j * np.random.randn(l_sketch, N),
            np.random.randn(N, l_sketch) + 1j * np.random.randn(N, l_sketch))
train_nonpar = lambda L, center, radius: beyn(L, center, radius, *lhs_rhs(L),
                                              200, 1e-10, 5)
tol = 1e-2 / radius # rescale by radius
interp_kind, patch_width = "spline3", 7

# train
model, ps_train = train(L, train_nonpar, 0., 1., interp_kind,
                        patch_width, p_range, tol)

# test
ps = np.linspace(*p_range, 200)
ps_coarse = ps[::10]
getApprox = lambda p: evaluate(model, ps_train, p, 0., 1., interp_kind, patch_width)
def getExact(p): # reference solution
    v_ref = train_nonpar(lambda z: L(z, p), 0., 1.)
    return np.sort(v_ref[np.abs(v_ref) <= 1.])
val_app, val_ref, error = runTest(ps, 10, getApprox, getExact)
# shift and rescale to original range
val_app, val_ref = val_app * radius + center, val_ref * radius + center
error *= radius

# plot approximation and error
plt.figure(figsize = (15, 5))
plt.subplot(141)
plt.plot(np.real(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 0]), ps, 'b:')
plt.plot(np.real(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.real(val_app[:, 1:]), ps, 'b:')
plt.legend(['exact', 'approx'])
plt.xlim(3, 7), plt.ylim(*p_range)
plt.xlabel("Re(lambda)"), plt.ylabel("p")
plt.subplot(142)
plt.plot(np.imag(val_ref[:, 0]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
plt.plot(np.imag(val_ref[:, 1:]), ps_coarse, 'ro')
plt.plot(np.imag(val_app[:, 1:]), ps, 'b')
plt.legend(['exact', 'approx'])
plt.xlim(-1, 0), plt.ylim(*p_range)
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
