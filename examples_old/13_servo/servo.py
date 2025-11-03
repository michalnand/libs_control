import numpy
import LibsControl


class Servo:
    def __init__(self, k, tau, dt = 0.001):

        
        # create dynamical system
        # dx = Ax + Bu
        # servo model, 2nd order
        # state x = (position, angular velocity)
        self.dt = dt

        mat_a = numpy.zeros((2, 2))
        mat_b = numpy.zeros((2, 1))

        mat_a[0][1] = 1.0
        mat_a[1][1] = -1.0/tau

        mat_b[1][0] = k/tau

        #create dynamical system
        self.ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

        self.x0 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0   
        self.x3 = 0.0


    def step(self, u_in):
        self.x3 = self.x2
        self.x2 = self.x1
        self.x1 = self.x0

        u = numpy.zeros((1, 1))
        u[0, 0] = u_in
        
        x, y = self.ds.forward_state(u)

        self.x0 = x[0, 0]

        pos     = self.x0
        vel     = (self.x0 - self.x1)/self.dt
        acc     = (self.x0 - 2.0*self.x1 + self.x2)/(self.dt**2)
        jerk    = (self.x0 - 3.0*self.x1 + 3.0*self.x2 - self.x3)/(self.dt**3)

        x        = numpy.array([[pos, vel, acc, jerk]]).T

        return x
