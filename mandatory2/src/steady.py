from dolfin import *
from mshr import *
import numpy as np
from fenics import *
set_log_active(False)

def mmesh(elements_in_rectangle, points_on_cylinder, mesh_name):
    """
        Creates a rectangular duct with dimensions
        L = 2.2 and H = 0.41. Inside is a cylinder
        placed at (0.2, 0.2) with a radius r=0.05
    """
    """
    rectangle = Rectangle(Point(0.0, 0.0), Point(2.2, 0.41))
    circle = Circle(Point(0.2, 0.2), 0.05, points_on_cylinder)
    domain = rectangle - circle
    mesh = generate_mesh(domain, elements_in_rectangle)
    """

    L = 2.2
    H = 0.41
    x0, y0 = 0.2, 0.2
    cylinder_radius = 0.05
    
    mesh = Mesh(mesh_name)
    #wiz = plot(mesh, interactive=False)
    #wiz.write_png("mesh2a")

    mf = FacetFunction('size_t', mesh)
    mf.set_all(0)

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

    walls = Walls()
    walls.mark(mf, 1)
    inlet = Inlet()
    inlet.mark(mf, 2)
    outlet = Outlet()
    outlet.mark(mf, 3)
    cylinder = Cylinder()
    cylinder.mark(mf, 4)

    #plot(mf, interactive=True)

    return mesh, mf


def boundary_conditions(mesh, mf, middle_velocity, W):
    """
        Returns noslip on the walls and on the cylinder.
        Inlet velocity is a fixed equation but is dependent
        on the middle velocity which must be given as a argument.
        Outlet conditions are set as neumann conditions
        for the velocity. This is default in FEniCS and
        are hence not included as a BC.
    """
    
    inlet_velocity = Expression((('4.0*U*x[1]*(H - x[1]) / pow(H, 2)', '0')), U=middle_velocity, H=0.41)

    wallsBC = DirichletBC(W.sub(0), Constant((0, 0)), mf, 1)
    cylinderBC = DirichletBC(W.sub(0), Constant((0, 0)), mf, 4)
    inletBC = DirichletBC(W.sub(0), inlet_velocity, mf, 2)

    BCs = [wallsBC, cylinderBC, inletBC]
    
    return BCs
    


def steady_solver(middle_velocity, mesh_name):
    """
        Solves the steady Navier-Stokes equation
        for an incompressible newtonian fluid.
        Returns the velocity and pressure components.
    """
    nu = 0.001
    rho = 1
    mesh, mf = mmesh(100, 300, mesh_name)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    P = FunctionSpace(mesh, 'CG', 1)
    W = MixedFunctionSpace([V, P])

    BCs = boundary_conditions(mesh, mf, middle_velocity, W)

    up = Function(W)
    u, p = split(up)
    
    vq = TestFunction(W)
    v, q = split(vq)

    eq1 = nu*inner(grad(u), grad(v))*dx - inner(p, div(v))/rho*dx + inner(dot(u, nabla_grad(u)), v)*dx # dot(dot(u,grad), u)
    eq2 = inner(div(u), q)*dx

    equation = eq1 - eq2

    solve(equation == 0, up, BCs)

    u, p = split(up)
    #plot(p)
    #plot(u)
    #interactive(True)

    #wiz1 = plot(u, interactive=False)
    #wiz2 = plot(p, interactive=False)
    #wiz1.write_png("u2a")
    #wiz2.write_png("p2a")
    return u, p, mesh, mf, rho, nu, W, V, P


def Coefficients(middle_velocity, solver, mesh_name):    
    """
        Calculates drag and lift coefficients.
        The dynamic viscosity is set to rho*nu.
    
        Calculates the pressure difference
        between the front and back of the cylinder
    """

    if solver == 'steady':
        u, p, mesh, mf, rho, nu, W, V, P = steady_solver(middle_velocity, mesh_name)
    else:
        0

    # number of unknowns
    Vdim = V.dim()
    Pdim = P.dim()
    Wdim = W.dim()
    print ''
    print 'Unknown velocity components:',Vdim
    print 'Unknown pressure components:',Pdim
    print 'Total number of unknowns:',Wdim

    Umean = 2.0*middle_velocity / 3
    cylinder_diameter = 0.1

    n = FacetNormal(mesh)
    ds = Measure('ds', subdomain_data=mf)

    # cauchys stress tensor
    tau = -p*Identity(2) + rho*nu*(grad(u) + grad(u).T)

    force = dot(tau, n)
    
    Drag = -assemble(force[0]*ds(4))
    Lift = -assemble(force[1]*ds(4))

    CD = 2*Drag / (rho*Umean**2*cylinder_diameter)
    CL = 2*Lift / (rho*Umean**2*cylinder_diameter)
    print ''
    print 'Drag coefficient:',CD
    print 'Lift coefficient:',CL

    # calculating pressure diffence 
    # between front and back of cylinder
    xa = np.array([0.15, 0.2])
    xe = np.array([0.25, 0.2])
    delta_pressure = p(xa) - p(xe)

    print ''
    print 'The pressure difference between back and front of cylinder:',delta_pressure
    
    x = [0.2501 + 0.0001*i for i in range(10000)]
    for i in x:
        velocity_x = u[0]([i,0.2])
        if velocity_x >= 0:
            xr = i
            break
    length_recirculation = xr - xe[0]
    print ''
    print 'Length of recirculation zone:',length_recirculation


"""
Meshes = ['mesh/von_karman_steady_coarse.xml', 'mesh/von_karman_steady_medium.xml', 'mesh/von_karman_steady_fine.xml']

for meshes in Meshes:
    Coefficients(0.3, 'steady', meshes)
"""
steady_solver(0.3, 'mesh/von_karman_steady_coarse.xml')

