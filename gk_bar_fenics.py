# u(0) = 0 and x = L is free.
# legacy FEniCS (dolfin 2019): python3 gk_bar_fenics.py
import numpy as np
from dolfin import *
 
set_log_level(LogLevel.ERROR)
 
# parameters
C1, C2 = 1.0, 10.0
alpha = 0.25            # C2 layer is alpha < y < beta
beta = 1.0 - alpha
ell = beta - alpha      # thickness of the C2 layer
h0, b0, L = 0.8, 1.0, 1.0
N_plot = 10             # cells of the exact bar
y0 = 0.5                # reference point in P
 
def h(x):
    return h0 * np.sin(np.pi * x / L)
 
 
def C_of(y):
    if alpha < y < beta:
        return C2
    return C1
 
def make_mesh(pts):
    # interval mesh with vertices exactly at pts (sorted)
    mesh = Mesh()
    ed = MeshEditor()
    ed.open(mesh, "interval", 1, 1)
    ed.init_vertices(len(pts))
    for i, p in enumerate(pts):
        ed.add_vertex(i, np.array([p]))
    ed.init_cells(len(pts) - 1)
    for i in range(len(pts) - 1):
        ed.add_cell(i, np.array([i, i + 1], dtype=np.uintp))
    ed.close()
    return mesh
 
def subdivide(key, m):
    key = np.unique(np.round(key, 14))
    return np.unique(np.concatenate([np.linspace(a, b, m + 1) for a, b in zip(key[:-1], key[1:])]))
 
def modulus(mesh, scale=1.0):
    V0 = FunctionSpace(mesh, "DG", 0)
    C = Function(V0)
    vals = C.vector().get_local()
    for c in cells(mesh):
        y = (c.midpoint().x() / scale) % 1.0
        vals[V0.dofmap().cell_dofs(c.index())[0]] = C_of(y)
    C.vector().set_local(vals)
    return C
 
class PeriodicY(SubDomain):
    # y = 1 is the same point as y = 0
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0.0) and on_boundary)
 
    def map(self, x, y):
        y[0] = x[0] - 1.0
 
# cell problems
def solve_cell(hx, m=8, full=False):
    # A_eff, D_eff and P at a point where the forces are hx apart
    # (full=True also gives back psi and C, for the psi plot at the end)
    ym, yp = 0.5 - hx / 2, 0.5 + hx / 2
    mesh = make_mesh(subdivide([0.0, 0.5 * alpha, alpha, ym, yp, beta, 1.0], m))
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicY())
    C = modulus(mesh)
    w, z = TrialFunction(V), TestFunction(V)
    # chi and psi are only unique up to a constant, so pin them at y = alpha/2
    pin = DirichletBC(V, 0.0, "near(x[0], 0.5*(%g))" % alpha, method="pointwise")
 
    # chi:  int C chi' z' = -int C z'
    K = assemble(C * w.dx(0) * z.dx(0) * dx)
    r = assemble(-C * z.dx(0) * dx)
    pin.apply(K, r)
    chi = Function(V)
    solve(K, chi.vector(), r)
    A_eff = assemble(C * (1.0 + chi.dx(0)) * dx)
 
    # psi:  int C psi' z' = b0 z(y+) - b0 z(y-)
    K = assemble(C * w.dx(0) * z.dx(0) * dx)
    r = assemble(Constant(0.0) * z * dx)
    if hx > 0:
        PointSource(V, Point(yp), b0).apply(r)
        PointSource(V, Point(ym), -b0).apply(r)
    pin.apply(K, r)
    psi = Function(V)
    solve(K, psi.vector(), r)
    D_eff = -assemble(C * psi.dx(0) * dx)
 
    P = b0 * ((yp - y0) - (ym - y0))   # int (y - y0) b_{-1} dy
    if full:
        return A_eff, D_eff, P, psi, C
    return A_eff, D_eff, P
 
# exact bar
def solve_exact(N, m=1):
    # N cells
    eps = L / N
    k = np.arange(N)
    xk = (k + 0.5) * eps               # cell centers, h is taken there
    xm = (k + 0.5 - h(xk) / 2) * eps   # -b0
    xp = (k + 0.5 + h(xk) / 2) * eps   # +b0
    key = np.r_[0.0, L, k * eps, (k + alpha) * eps, (k + beta) * eps, xm, xp]
    mesh = make_mesh(subdivide(key, m))
    V = FunctionSpace(mesh, "CG", 1)
    C = modulus(mesh, scale=eps)
    u, v = TrialFunction(V), TestFunction(V)
    K = assemble(C * u.dx(0) * v.dx(0) * dx)
    r = assemble(Constant(0.0) * v * dx)
    for a, b in zip(xm, xp):
        PointSource(V, Point(b), b0).apply(r)
        PointSource(V, Point(a), -b0).apply(r)
    DirichletBC(V, 0.0, "near(x[0], 0.0)").apply(K, r)   # only x = 0 is fixed
    uh = Function(V)
    solve(K, uh.vector(), r)
    xv = np.sort(mesh.coordinates()[:, 0])
    return xv, np.array([uh(Point(x)) for x in xv])
 
# homogenized bar
mesh_M = UnitIntervalMesh(200)
mesh_M.coordinates()[:] *= L
VM = FunctionSpace(mesh_M, "CG", 2)
x_dofs = VM.tabulate_dof_coordinates()[:, 0]
 
# one cell problem per macro dof
res = np.array([solve_cell(h(x)) for x in x_dofs])
A_all, D_all, P_all = res.T
A_eff = float(np.mean(A_all))   # it's the same everywhere
D_fun, P_fun = Function(VM), Function(VM)
D_fun.vector().set_local(D_all)
P_fun.vector().set_local(P_all)
 
def solve_homog(source):
    # int A_eff u0' v' = int source v',  u0(0) = 0, nothing at x = L (free end)
    u, v = TrialFunction(VM), TestFunction(VM)
    u0 = Function(VM)
    solve(A_eff * u.dx(0) * v.dx(0) * dx == source * v.dx(0) * dx, u0,
          DirichletBC(VM, 0.0, "near(x[0], 0.0)"))
    return u0
 
 
sols = {"D+P": solve_homog(D_fun + P_fun),
        "P only": solve_homog(P_fun),
        "periodic": solve_homog(D_fun)}
 
# check against the closed forms
A_cf = 1.0 / ((1 - ell) / C1 + ell / C2)   # harmonic mean
 
def J(hh):
    return np.where(hh <= ell, hh / C2, ell / C2 + (hh - ell) / C1)
 
D_cf = A_cf * b0 * J(h(x_dofs)) - b0 * h(x_dofs)
 
# free end: u0(L) = int_0^L b0 J(h) dx
uL = 2 * b0 * h0 * L / (np.pi * C2)
if h0 > ell:  
    uL += b0 * L * (1 / C1 - 1 / C2) * (2 / np.pi * np.sqrt(h0**2 - ell**2)
                                        - ell * (1 - 2 / np.pi * np.arcsin(ell / h0)))
uL_cf = {"D+P": uL, "P only": 2 * b0 * h0 * L / (np.pi * A_cf)}
uL_cf["periodic"] = uL_cf["D+P"] - uL_cf["P only"]
 
print("A_eff  fenics %.12f  closed %.12f  (spread %.1e)" % (A_eff, A_cf, np.ptp(A_all)))
print("max |D_eff - closed| = %.1e" % np.max(np.abs(D_all - D_cf)))
print("u0(L)")
for name, u0 in sols.items():
    print("  %-9s fenics % .10f  closed % .10f" % (name, u0(Point(L)), uL_cf[name]))
print("exact bar")
for N in [10, 40, 160]:
    xe, ue = solve_exact(N)
    err = max(abs(ue[i] - sols["D+P"](Point(xe[i]))) for i in range(len(xe)))
    print("  N = %3d  u(L) = %.10f  max|u - u0| = %.2e" % (N, ue[-1], err))
 
# plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
xe, ue = solve_exact(N_plot)
xf = np.linspace(0, L, 401)
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(xf, [sols["D+P"](Point(x)) for x in xf], color="0.7", lw=4,
        label=r"source $D^{\mathrm{eff}}+P$")
ax.plot(xe, ue, color="k", lw=0.9, label=rf"exact, $N={N_plot}$")
ax.plot(xf, [sols["P only"](Point(x)) for x in xf], "k--", lw=1.4, label=r"source $P$")
ax.plot(xf, [sols["periodic"](Point(x)) for x in xf], "k:", lw=1.6,
        label=r"source $D^{\mathrm{eff}}$")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, L)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("fig_gk_u.png", dpi=200)
print("saved fig_gk_u.png")
 
idx = np.argsort(x_dofs)  
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(x_dofs[idx], P_all[idx], "k--", lw=1.4, label=r"$P$")
ax.plot(x_dofs[idx], D_all[idx] + P_all[idx], "k-", lw=1.2, label=r"$D^{\mathrm{eff}}+P$")
ax.plot(x_dofs[idx], D_all[idx], "k:", lw=1.6, label=r"$D^{\mathrm{eff}}$")
if h0 > ell:
    # between these two lines the forces are in C1
    x_star = L / np.pi * np.arcsin(ell / h0)
    for xx in (x_star, L - x_star):
        ax.axvline(xx, color="0.6", lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, L)
ax.set_xlabel(r"$x$")
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.48), fontsize=10)
fig.tight_layout()
fig.savefig("fig_gk_P.png", dpi=200)
print("saved fig_gk_P.png")
 
dmin = D_all.min()
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(x_dofs[idx], D_all[idx], "k-", lw=1.4, label=r"cell problems (FEniCS)")
ax.plot(x_dofs[idx][::10], D_cf[idx][::10], "o", mfc="white", mec="k", ms=5,
        label=r"closed form, $A^{\mathrm{eff}}b_0J(h)-b_0h$")
if h0 > ell:
    for xx in (x_star, L - x_star):
        ax.plot([xx, xx], [1.25 * dmin, -0.15 * dmin], color="0.5", lw=0.8, ls="--")
    ax.text(x_star / 2, -0.05 * dmin, r"forces in $C_2$", ha="center", fontsize=9)
    ax.text(L / 2, -0.05 * dmin, r"forces in $C_1$", ha="center", fontsize=9)
    ax.text(L - x_star / 2, -0.05 * dmin, r"forces in $C_2$", ha="center", fontsize=9)
    ax.text(x_star + 0.01, 1.18 * dmin, r"$x^*$", ha="left", fontsize=10)
    ax.text(L - x_star - 0.01, 1.18 * dmin, r"$L-x^*$", ha="right", fontsize=10)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, L)
ax.set_ylim(1.25 * dmin, -0.5 * dmin)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$D^{\mathrm{eff}}$")
ax.legend(frameon=False, loc="upper center", fontsize=9, ncol=2, columnspacing=1.2)
fig.tight_layout()
fig.savefig("fig_gk_Deff.png", dpi=200)
print("saved fig_gk_Deff.png")
 
# D_eff at that x is minus the area under C psi'.
x_star = L / np.pi * np.arcsin(min(ell / h0, 1.0))
x_show = [0.1, x_star, 0.3, 0.5]
names = [r"$x=0.1$", r"$x=x^*$", r"$x=0.3$", r"$x=0.5$"]
styles = ["-", "--", "-.", ":"]
fig1, ax1 = plt.subplots(figsize=(4.2, 3.2))
fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
for ax in (ax1, ax2):
    ax.axvspan(alpha, beta, color="0.9", lw=0)   # C2 layer
for x, name, ls in zip(x_show, names, styles):
    A_x, D_x, P_x, psi, C = solve_cell(h(x), full=True)
    mesh = psi.function_space().mesh()
    y = np.sort(mesh.coordinates()[:, 0])
    ax1.plot(y, [psi(Point(yy)) for yy in y], color="k", ls=ls, lw=1.3, label=name)
    V0 = FunctionSpace(mesh, "DG", 0)
    S = project(C * psi.dx(0), V0)
    ys, ss = [], []
    for c in sorted(cells(mesh), key=lambda c: c.midpoint().x()):
        a, b = sorted(c.get_vertex_coordinates())
        val = S.vector()[V0.dofmap().cell_dofs(c.index())[0]]
        ys += [a, b]
        ss += [val, val]
    ax2.plot(ys, ss, color="k", ls=ls, lw=1.3, label=name + r", $D^{\mathrm{eff}}=%.3f$" % D_x)
ax1.set_ylabel(r"$\psi$")
ax2.set_ylabel(r"$C\,\partial\psi/\partial y$")
for ax in (ax1, ax2):
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$y$")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax2.legend(frameon=False, fontsize=8, loc="lower center", handlelength=1.6)
for f, name in ((fig1, "fig_gk_psi.png"), (fig2, "fig_gk_stress.png")):
    f.tight_layout()
    f.savefig(name, dpi=200)
    print("saved", name)
