# elliptical section

# constants

dpdx = -100
mu = 1
a = 1
b = 0.8


from dolfin import *
from mshr import *
import numpy as np
set_log_active(False)

M = [10, 20, 40, 80]
O = [1]#, 2, 3]

domain = Ellipse(Point(0,0), a, b, 100)

for o in O:
    print ''
    print 'Polynomial order:',o
    error = np.zeros(len(M))  
    #mesh = generate_mesh(domain, 1)
    for m in range(len(M)):
        mesh = generate_mesh(domain, M[m])
        #mesh = refine(mesh)
        V = FunctionSpace(mesh, 'CG', o)

        class walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        wall = walls()
        mf = FacetFunction("size_t", mesh)
        mf.set_all(2)
        wall.mark(mf, 1)
        noslip = DirichletBC(V, Constant(0), mf, 1)
        plot(mf, interactive=True)


        u = TrialFunction(V)
        v = TestFunction(V)
        c = -inner(grad(u), grad(v))*mu*dx
        L = dpdx*v*dx

        u_ = Function(V)
        solve(c == L, u_, noslip)

        # exact solution
            
        ue = Expression('-1./(2*mu)*dpdx*a*a*b*b/(a*a+b*b)*(1 - x[0]*x[0]/(a*a) - x[1]*x[1]/(b*b))', dpdx=dpdx, mu=mu, a=a, b=b)
        uex = project(ue, V, noslip)

        Qe = np.pi*(-dpdx)*a*a*a*b*b*b / (4*mu*(a*a + b*b))

        flux = u_*dx
        total_flux = assemble(flux)
        #print total_flux, Qe
        flux_error = abs(Qe - total_flux)

        error[m] = errornorm(u_, uex)
        if m == 0:
            print 'Mesh size: %0.6g, Errornorm: %.8g' %(mesh.hmin(), error[m])
            print 'Flux error:', flux_error
        else:
            print 'Mesh size: %0.6g, Errornorm: %.8g, rate: %.5g'% \
            (mesh.hmin(), error[m], -np.log((error[m]/error[m-1]))/np.log((M[m]/M[m-1])))
            print 'Flux error:',flux_error
    
#uu = File("ex1_47_num.pvd")
#uu << u_


