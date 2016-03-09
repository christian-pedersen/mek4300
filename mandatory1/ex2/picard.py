# solve eq(148) and eq(150) using picard iteration

from dolfin import *
set_log_active(False)

L = 4

mesh = IntervalMesh(100, 0, L)
V = FunctionSpace(mesh, 'CG', 1)
V2 = FunctionSpace(mesh, 'CG', 4)
W = MixedFunctionSpace([V, V])
#W2 = MixedFunctionSpace([V2, V2])

def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def right(x, on_boundary):
    return near(x[0], L) and on_boundary

BC1 = DirichletBC(W.sub(0), 0, left)
BC2 = DirichletBC(W.sub(1), 0, left)
BC3 = DirichletBC(W.sub(0), 1, right)


HF = TrialFunction(W)
H, F = split(HF)

HFt = TestFunction(W)
Ht, Ft = split(HFt)

HF_ = Function(W)
H_, F_ = split(HF_)


k = 0
K = 100
eps = 1e-14
error = 1

HF_1 = interpolate(Expression(('0', 'x[0]')), W)
H_1, F_1 = split(HF_1)

H1 = -inner(grad(H), grad(Ht))*dx +   F_1*H.dx(0)*Ht*dx + Ht*dx - H*H_1*Ht*dx
H2 = -inner(grad(H), grad(Ht))*dx + 2*F_1*H.dx(0)*Ht*dx + Ht*dx - H*H_1*Ht*dx
H3 = H*Ft*dx - F.dx(0)*Ft*dx

H13 = H1 + H3
H23 = H2 + H3


while k < K and error > eps:
    solve(lhs(H23) == rhs(H23), HF_, [BC1, BC2, BC3])
    error = errornorm(HF_, HF_1)
    HF_1.assign(HF_)
    print "k = ",k,", Error = ", error
    k += 1

plot(F_, interactive=True)
#wiz = plot(F_, interactive=False)
#wiz.write_png("ex2_picard_2")
