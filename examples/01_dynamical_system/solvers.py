import numpy
import matplotlib.pyplot as plt


def forward_func(x, u):
    dx = A@x + B@u
    y  = C@x 

    return dx, y

def ODESolverRK4(forward_func, x, u, dt):
    k1, y1  = forward_func(x, u) 
    k1      = k1*dt  

    k2, y2  = forward_func(x + 0.5*k1, u + 0.5*dt)
    k2      = k2*dt

    k3, y3  = forward_func(x + 0.5*k2, u + 0.5*dt)
    k3      = k3*dt

    k4, y4  = forward_func(x + k3, u + dt)
    k4      = k4*dt

    x_new   = x + (1.0/6.0)*(1.0*k1 + 2.0*k2 + 2.0*k3 + 1.0*k4)
    y       = (1.0/6.0)*(1.0*y1 + 2.0*y2 + 2.0*y3 + 1.0*y4)
    
    return x_new, y




def simulate(n, m, dt, n_steps):
    # input and 
    u = numpy.zeros((m, 1))
    x = numpy.zeros((n, 1))

    # log results
    t_result = []
    u_result = []
    x_result = []
    y_result = []

    for n in range(n_steps):
        # solver step
        x, y = ODESolverRK4(forward_func, x, u, dt)
   
        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        x_result.append(x[:, 0])
        y_result.append(y[:, 0])

    t_result = numpy.array(t_result)
    u_result = numpy.array(u_result)
    x_result = numpy.array(x_result)    
    y_result = numpy.array(y_result)  

    return t_result, u_result, x_result, y_result



if __name__ == "__main__":
    dt = 0.01
    n_steps = 1000
    n = 12
    m = 4

    A = numpy.random.randn(n, n)
    B = numpy.random.randn(n, m)
    C = numpy.eye(n)

    simulate(n, m, dt, n_steps)
