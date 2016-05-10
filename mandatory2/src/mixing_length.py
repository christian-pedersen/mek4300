from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

H = 1.0
N = 501
Re = 1000
vstar = 0.05
nu = vstar/Re
kappa = 0.41
A = 26


mesh = IntervalMesh(N, 0, H/2)
V = FunctionSpace(mesh, 'CG', 1)

u = Function(V)
v = TestFunction(V)

yp = Expression('0.05*x[0]/nu', nu=nu)
l = Expression('kappa*x[0]*(1-exp(-yp/A))', kappa=kappa, yp=yp, A=A)
dpdx = Constant(2*0.05*0.05/H)


class walls(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS

boundaries = FacetFunction('size_t', mesh)
walls = walls()
walls.mark(boundaries, 2)

noslip = DirichletBC(V, Constant(0.0), boundaries, 2)

eq1 = -nu*inner(u.dx(0), v.dx(0))*dx + dpdx*v*dx - l*l*inner(abs(u.dx(0))*u.dx(0), v.dx(0))*dx

solve(eq1 == 0, u, noslip)


def exerciseA():
    y = np.linspace(0, 0.5, len(u.vector().array()[:]))
    u_array = [i for i in u.vector().array()[::-1]]
    plt.plot(y, u.vector().array()[::-1], label='velocity profile')
    plt.title('Velocity profile for a half channel height')
    plt.grid('on')
    plt.xlabel('channel height [m]')
    plt.legend(loc='lower right')
    plt.ylabel('u(y)')
    plt.show()


def exerciseB(kappa, B):
    y = np.linspace(0, 0.5, len(u.vector().array()[:]))
    u_array = [i for i in u.vector().array()[::-1]]
    plt.plot(y, u_array, label='numerical velocity')

    y5 = 50*y[:]
    i = 0
    while y5[i]/0.05 < 5:
        i += 1
    plt.plot(y[0:i], y5[0:i], 'o', label='u+ for y+<5')

    y30 = [0.05/kappa*np.log(1000*y[i]) + 0.05*B for i in range(len(y))]
    print y30
    j = 0
    while (1000*y[j]) < 30:
        j += 1
    plt.plot(y[j:-1], y30[j:-1], '--', label='u+ for y+>30')
    plt.title('K=%g, B=%g'%(kappa, B))
    plt.legend(loc='lower right')
    plt.xlabel('channel height [m]'), plt.ylabel('u(y)')
    plt.grid('on')
    plt.show()



#exerciseA()
#kappa = [0.41, 0.41, 0.43]
#B = [5.5, 5., 5.5]
#for i in range(len(kappa)):
#    exerciseB(kappa[i], B[i])

