'''
Euler solver, 1st order
'''
def ODESolverEuler(dynamical_system, x, u, dt = 0.001):
    dx, y  = dynamical_system(x, u)
    return x + dx*dt, y

'''
Ruge-Kuta solver, 4th order
'''
def ODESolverRK4(dynamical_system, x, u, dt = 0.001):
    k1, y1  = dynamical_system(x, u)
    k1      = k1*dt  

    k2, y2  = dynamical_system(x + 0.5*k1, u + 0.5*dt)
    k2      = k2*dt

    k3, y3  = dynamical_system(x + 0.5*k2, u + 0.5*dt)
    k3      = k3*dt

    k4, y4  = dynamical_system(x + k3, u + dt)
    k4      = k4*dt

    x_new   = x + (1.0/6.0)*(1.0*k1 + 2.0*k2 + 2.0*k3 + 1.0*k4)
    y       = (1.0/6.0)*(1.0*y1 + 2.0*y2 + 2.0*y3 + 1.0*y4)
    
    return x_new, y