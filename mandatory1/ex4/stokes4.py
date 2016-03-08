# compute numerical solution for laminar Stokes flow

from dolfin import *
import os
import matplotlib.pyplot as plt
set_log_level(WARNING)

velocity = -1
mu = 100
L = 1

#os.system("dolfin-convert ex4_mesh.msh ex4_mesh.xml mesh/")
mesh = Mesh("mesh/ex4_mesh.xml")

V = VectorFunctionSpace(mesh, 'CG', 2)
P = FunctionSpace(mesh, 'CG', 1)

VP = MixedFunctionSpace([V, P])

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

#print mesh.hmin()

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > L - DOLFIN_EPS and on_boundary

class Plate(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.5*L - DOLFIN_EPS and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.1*L) or near(x[1], 0) and on_boundary

class Step(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.5*L) and on_boundary)

def stream_function(u):
    U = FunctionSpace(mesh, 'CG', 2)#u.function_space().sub(0).collapse()
    psi = TrialFunction(U)
    phi = TestFunction(U)
    grad_psi = as_vector((-u[1], u[0]))
    n = FacetNormal(mesh)

    a = - inner(grad(psi), grad(phi))*dx 
    L = - dot(grad_psi, n)*phi*ds + inner(-u[1].dx(0) + u[0].dx(1), phi)*dx 
    
    #BC = DirichletBC(U, Constant('0'), "on_boundary")
    #def lower(x, on_boundary):
    #    return near(x[1], 0) and on_boundary
    #BC = DirichletBC(U, Constant(0), lower)
    G = a - L
    psi = Function(U)
    
    solve(a == L, psi)
    return psi, U


inlet = Inlet()
outlet = Outlet()
inlet.mark(mf, 8)
outlet.mark(mf, 9)

plate = Plate()
plate.mark(mf, 2)

walls = Walls()
walls.mark(mf, 1)
step = Step()
step.mark(mf, 1)
#plot(mf, interactive=True)

noslip = DirichletBC(VP.sub(0), Constant((0,0)), mf, 1)
moving_plate = DirichletBC(VP.sub(0), Constant((velocity,0)), mf, 2)

BCs = [noslip, moving_plate]

u, p = TrialFunctions(VP)
ut, pt = TestFunctions(VP)

a = -mu*inner(grad(u), grad(ut))*dx + inner(p, div(ut))*dx
L = inner(div(u), pt)*dx

aL = a - L

up_ = Function(VP)

solve(lhs(aL) == rhs(aL), up_, BCs)
u_, p_ = split(up_)

def center_vortex(u):
    psi, U = stream_function(u_)
    normalize(psi.vector())

    #psi_min = min(psi.vector().array())
    if velocity == -1:
        pme = psi.vector().array().argmax()
    elif velocity == 1:
        pme = psi.vector().array().argmin()
    xcoor = interpolate(Expression('x[0]'), U)
    ycoor = interpolate(Expression('x[1]'), U)
    print 'Coordinates for center of vortex: '
    print 'x = %g, y = %g' %(xcoor.vector()[pme], ycoor.vector()[pme])


def flux(u):
    n = FacetNormal(mesh)
    ds = Measure('ds', subdomain_data=mf)
    flux_inn = assemble(dot(u,-n)*ds(8))
    flux_out = assemble(dot(u,n)*ds(9))
    print 'Velocity flux inn:',flux_inn
    print 'Velocity flux out:',flux_out
    print 'Flux error:', flux_inn-flux_out


def normal_stress(p):
    n = FacetNormal(mesh)
    ds = Measure('ds', subdomain_data=mf)
    NS = assemble(-p*ds(1))
    print 'Normal stress:', NS


#normal_stress(p_)
#center_vortex(u_)
#flux(u_)


