import numpy
import LibsControl

# reaction wheel inverted pendulum
class ReactionWheelInvertedPendulum:

    def __init__(self, m, l, Ip, Iw, g = 9.81, dt = 0.01):
        self.a = m*g*l/Ip
        self.b = 1.0/Ip
        self.c = 1.0/Iw

        self.dt = dt

        self.reset()
        
    def reset(self):
        self.x  = numpy.zeros((4, 1))
        self.u  = numpy.zeros((1, 1))

    def step(self, x, u):   
        x_new, _ = LibsControl.ode_solver.ODESolverRK4(self.forward_func, x, u, self.dt)
        return x_new


    def forward_func(self, x, u):

        dx = numpy.zeros((4, 1))    

        # pendulum, theta angle and angular rate
        dx[0, 0] = x[1, 0]
        dx[1, 0] = self.a*numpy.sin(x[0, 0]) - self.b*u[0, 0]

        # wheel, phi angle and angular rate
        dx[2, 0] = x[3, 0]
        dx[3, 0] = -self.a*numpy.sin(x[0, 0]) + self.c*u[0, 0]

        return dx, x

