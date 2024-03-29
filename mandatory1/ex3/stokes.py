# compute numerical solution for laminar Stokes flow
# for a driven cavity.

from dolfin import *
set_log_level(WARNING)

mu = 100

mesh = UnitSquareMesh(100, 100)
V = VectorFunctionSpace(mesh, 'CG', 2)
P = FunctionSpace(mesh, 'CG', 1)

VP = MixedFunctionSpace([V, P])

mf = FacetFunction("size_t", mesh)
mf.set_all(2)


def stream_function(u):
    U = FunctionSpace(mesh, 'CG', 2) #V = u.function_space().sub(0).collapse()
    psi = TrialFunction(U)
    phi = TestFunction(U)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx  # dv/dx - du/dy
    bc = DirichletBC(U, Constant(0.), DomainBoundary())

    #A = assemble(a)
    #b = assemble(L)
    #bc.apply(A,b)
    A, b = assemble_system(a, L, bc)
    psi = Function(U)
    solve(A, psi.vector(), b)

    return psi, U

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0) or 
                near(x[0], 1) or 
                near(x[1], 0) and on_boundary)

class Plate(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary

wall = Walls()
wall.mark(mf, 3)
noslip = DirichletBC(VP.sub(0), Constant((0,0)), mf, 3)

plate = Plate()
plate.mark(mf, 1)
moving_plate = DirichletBC(VP.sub(0), Constant((1,0)), mf, 1)

BCs = [noslip, moving_plate]

#plot(mf, interactive=True)

u, p = TrialFunctions(VP)
ut, pt = TestFunctions(VP)

a = -mu*inner(grad(u), grad(ut))*dx + inner(p,div(ut))*dx
L = inner(div(u),pt)*dx

aL = a + L
up_ = Function(VP)

solve(lhs(aL) == rhs(aL), up_, BCs)
u_, p_ = up_.split()
psi, U = stream_function(u_)

#plot(psi, interactive=True)
#plot(p_)
#plot(psi)
#uu = File("ex3_u.pvd")
#uu << u_
#pp = File("ex3_p.pvd")
#pp << p_
#psipsi = File("ex3_psi.pvd")
#psipsi << psi
#interactive(True)

psi_min = min(psi.vector().array())
pme = psi.vector().array().argmin()
xcoor = interpolate(Expression('x[0]'), U)
ycoor = interpolate(Expression('x[1]'), U)
print 'Coordinates for center of vortex: '
print 'x = %g, y = %g' %(xcoor.vector()[pme], ycoor.vector()[pme])

