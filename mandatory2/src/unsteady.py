from dolfin import *
import numpy as np

# constants

L = 2.2
H = 0.41
rho = Constant(1.0)
nu = Constant(0.001)
mu = rho*nu
U_m = 1.5
x0, y0 = 0.2, 0.2
cylinder_radius = 0.05
U_ = 2.0*U_m / 3
cylinder_diameter = 0.1
xa = np.array([0.15, 0.2])
xe = np.array([0.25, 0.2])
dt = 0.0005
end_time = 8

# create mesh and boundaries
mesh = Mesh('mesh/von_karman_coarse.xml')

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) or near(x[1], H)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return (pow(x[0]-x0, 2) + pow(x[1]-y0, 2) - pow(cylinder_radius, 2) < DOLFIN_EPS) and on_boundary

boundaries = FacetFunction('size_t', mesh)
boundaries.set_all(0)

inlet = Inlet()
inlet.mark(boundaries, 1)
outlet = Outlet()
outlet.mark(boundaries, 2)
walls = Walls()
walls.mark(boundaries, 3)
cylinder = Cylinder()
cylinder.mark(boundaries, 4)

plot(boundaries, interactive=True)

# function spaces and functions

V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

u0 = Function(V)
p0 = Function(Q)
ustar = Function(V)
u1 = Function(V)
p1 = Function(Q)

uhalf = 0.5*(u0+u)

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
def sigma(u, p):
    """
        Cauchy stress tensor
    """
    return 2*mu*epsilon(u) - p*Identity(len(u))

n = FacetNormal(mesh)

# boundary conditions

inlet_velocity = Expression(('4*U_m*x[1]*(0.41-x[1]) / (H*H)', '0'), H=H, U_m=U_m)
outlet_pressure = Constant(0)
solids_velocity = Constant((0, 0))

inletBC = DirichletBC(V, inlet_velocity, inlet)
outletBC = DirichletBC(Q, outlet_pressure, outlet)
noslip_walls = DirichletBC(V, solids_velocity, walls)
noslip_cylinder = DirichletBC(V, solids_velocity, cylinder)


velocityBC = [noslip_walls, noslip_cylinder, inletBC]
pressureBC = [outletBC] 
"""
# equation for tentative velocity

eq1 = rho*dot((u-u0)/dt, v)*dx + rho*dot(dot(u0, grad(u0)), v)*dx \
    + inner(sigma(uhalf, p0), epsilon(v))*dx - mu*dot(grad(uhalf).T*n, v)*ds \
    + inner(p0*n, v)*ds

eq1 = rho*dot((u-u0)/dt, v)*dx + rho*dot(grad(u0)*u0, v)*dx \
     - mu*inner(grad(uhalf).T*n, v)*ds + dot(p0*n, v)*ds \
     + inner(sigma(uhalf, p0), epsilon(v))*dx
a1 = lhs(eq1)
L1 = rhs(eq1)

# equation for pressure correction

a2 = dt*dot(grad(p), grad(q))*dx
L2 = dt*dot(grad(p0), grad(q))*dx - rho*div(ustar)*q*dx

# equation for velocity correction

a3 = rho*inner(u1, v)*dx
L3 = rho*inner(ustar, v)*dx - dt*inner(grad(p1- p0), v)*dx

"""


# tentative velocity equation
eq1 = rho*dot((u-u0)/dt, v)*dx + rho*dot(grad(u0)*u0, v)*dx \
     - mu*inner(grad(uhalf).T*n, v)*ds + dot(p0*n, v)*ds \
     + inner(sigma(uhalf, p0), epsilon(v))*dx
a1 = lhs(eq1)
L1 = rhs(eq1)

# pressure correction equation
a2 = dt*dot(grad(p), grad(q))*dx
L2 = dt*dot(grad(p0), grad(q))*dx - rho*div(u1)*q*dx

# corrected velocity equation
a3 = rho*dot(u, v)*dx
L3 = rho*dot(u1, v)*dx + dot(dt*grad(p0-p1), v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

t = dt

_u = File('results_coarse/velocity.pvd')
_p = File('results_coarse/pressure.pvd')

pix = 0

results = open('results_coarse.txt', 'w')
results.write('t\tcd\tcl\tdp\n')

ds = Measure('ds', subdomain_data=boundaries)

while t < end_time +DOLFIN_EPS:
    print 'calculation progress:',(t-dt)/end_time*100,'%'

    # calculate tentative velocity
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in velocityBC]
    solve(A1, u1.vector(), b1)
    #solve(A1 == b1, ustar, velocityBC)

    # correct pressure
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in pressureBC]
    solve(A2, p1.vector(), b2)

    # correct velocity
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in velocityBC]
    solve(A3, u1.vector(), b3)


    # calculating drag/lift coefficients
    tau = -p1*Identity(2) + rho*nu*(grad(u1) + grad(u1).T)

    force = dot(tau, n)
    
    Drag = -assemble(force[0]*ds(4))
    Lift = -assemble(force[1]*ds(4))

    CD = 2*Drag / (1.0*(2.0/3.0*1.5)**2*0.1)#(rho*Umean**2*cylinder_diameter)
    CL = 2*Lift / (1.0*(2.0/3.0*1.5)**2*0.1) #(rho*Umean**2*cylinder_diameter)

    # calculating pressure diffence 
    # between front and back of cylinder

    delta_pressure = p1(xa) - p1(xe)


    if pix == 10:
        pix = 0
    if pix == 0:
        _u << u1
        _p << p1
    pix += 1
    t += dt

    results.write('%s\t%s\t%s\t%s\n' % (str(t), str(CD), str(CL), str(delta_pressure)))

    u0.assign(u1)
    p0.assign(p1)

results.close()
#plot(u1, interactive=True)


