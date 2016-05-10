from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

set_log_active(False)



def solver(beta, guess, L):

    mesh = IntervalMesh(100, 0, L)

    def left(x, on_boundary):
        return near(x[0], 0) and on_boundary

    def right(x, on_boundary):
        return near(x[0], L) and on_boundary


    V = FunctionSpace(mesh, 'CG', 1)
    W = MixedFunctionSpace([V, V])


    BC = [DirichletBC(W.sub(1), 0, left),
          DirichletBC(W.sub(0), 0, left),
          DirichletBC(W.sub(0), 1, right)]

    HF = interpolate(guess, W)
    H, F = split(HF)
    
    HFt = TestFunction(W)
    Ht, Ft = split(HFt)

    eq1 = -inner(grad(H), grad(Ht))*dx + F*H.dx(0)*Ht*dx + Constant(beta)*Ht*dx - Constant(beta)*H*H*Ht*dx
    eq2 = H*Ft*dx - F.dx(0)*Ft*dx

    equation = eq1 + eq2

    solve(equation == 0, HF, BC)

    return H, V, mesh


def a():

    Beta = [1.0, 0.3, 0.0, -0.1, -0.18, -0.198838]
    guess = Expression(('1', '1'))
    L = 6

    for beta in Beta:
        H, V, mesh = solver(beta, guess, L)
        Hdiv = project(H.dx(0), V)
        H = project(H, V)

        plt.figure(1)
        plt.plot(mesh.coordinates(), H.vector().array()[::-1], label='beta = %g' %beta)
        plt.title('Velocity profiles')
        plt.xlabel('$\eta$'), plt.ylabel("f'")
        plt.legend(loc='lower right')
        plt.grid('on')
        
        plt.figure(2)
        plt.plot(mesh.coordinates(), Hdiv.vector().array()[::-1], label='beta = %g' %beta)
        plt.title('Shear-stress profiles')
        plt.xlabel('$\eta$'), plt.ylabel("f''")
        plt.legend(loc='upper right')
        plt.grid('on')
    
    plt.show()


def b():
    L = 6
    beta = -0.1

    Guess = [Expression(('1', '1')), Expression(('x[1]', 'x[1]'))]
    for guess in Guess:
        H, V, mesh = solver(beta, guess, L)
        Hdiv = project(H.dx(0), V)
        if guess == Guess[0]:
            g = '[1, 1]'
        else:
            g = '[y, y]'
        plt.plot(mesh.coordinates(), Hdiv.vector().array()[::-1], label='guess = %s' %g)
        plt.title('Shear-stress profiles with $\ beta$ = -0.1')
        plt.xlabel('$\eta$'), plt.ylabel("f''")
        plt.legend(loc='lower right')
        plt.grid('on')

    plt.show()
#a()
b()



