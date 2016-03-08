# compare numerical vs analytical solution for equation 3-49

from dolfin import *
set_log_active(False)
import numpy as np
from mshr import *

M = [10, 20, 40, 80]
O = [1, 2, 3]
mu = 1
dpdx = -100

pts = [Point(0.5, -sqrt(3)/2), Point(0, 0), Point(-0.5, -sqrt(3)/2)]
shape = Polygon(pts)

class walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def Q_exact(a, mu, dpdx):
    return a**4*np.sqrt(3)*(-dpdx) / (320*mu)

Q = Q_exact(a=1, mu=mu, dpdx=dpdx)

for o in O:
    print ''
    print 'Polynomial order:',o
    error = np.zeros(len(M))
    mesh = generate_mesh(shape, 2)
    for m in range(len(M)):
        mesh = refine(mesh)
        V = FunctionSpace(mesh, 'CG', o)

        wall = walls()
        mf = FacetFunction("size_t", mesh)
        mf.set_all(2)
        wall.mark(mf, 1)
        noslip = DirichletBC(V, Constant(0), mf, 1)
        plot(mf, interactive=True)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = -inner(grad(u), grad(v))*mu*dx
        L = dpdx*v*dx

        u_ = Function(V)
        solve(a == L, u_, noslip)

        ue = Expression('-dpdx*( -x[1] - 0.5*a*sqrt(3) ) * ( 3*x[0]*x[0] - x[1]*x[1]) / (2*sqrt(3)*a*mu)', a=1, mu=mu, dpdx=dpdx)
        uex = project(ue, V, noslip)

        error[m] = errornorm(u_, uex)

        flux = u_*dx
        total_flux = assemble(flux)

        flux_error = abs(Q - total_flux)
        if m == 0:
            print 'Mesh size: %0.6g, Errornorm: %.8g' %(mesh.hmin(), error[m])
            print 'Flux error:', flux_error
        else:
            print 'Mesh size: %0.6g, Errornorm: %.8g, rate: %.5g'% \
            (mesh.hmin(), error[m], np.log((error[m-1]/error[m]))/np.log((M[m]/M[m-1])))
            print 'Flux error:', flux_error

#uu = File("ex1_49.pvd")
#uu << u_
