import numpy
import LibsControl


class ServoCL:
    def __init__(self, k, tau, q, r, dt = 0.001):

        
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

        a_disc, b_disc, c_disc = LibsControl.c2d(mat_a, mat_b, None, dt)

        print("model")
        print(a_disc)
        print(b_disc)
        print(q)
        print(r)

        self.lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r)


     

        self.reset()


    def reset(self):
        self.ds.reset()

        self.u = numpy.zeros((1, 1))

        self.x0 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0   
        self.x3 = 0.0


    def step(self, x_req):
        xr = numpy.zeros((2, 1))
        xr[0, 0] = x_req


        x = self.ds.x
        self.u = self.lqr.forward(xr, x, self.u)

        self.ds.forward_state(self.u)

        self.x3 = self.x2
        self.x2 = self.x1
        self.x1 = self.x0
        self.x0 = x[0, 0]

        pos     = self.x0
        vel     = (self.x0 - self.x1)/self.dt
        acc     = (self.x0 - 2.0*self.x1 + self.x2)/(self.dt**2)
        jerk    = (self.x0 - 3.0*self.x1 + 3.0*self.x2 - self.x3)/(self.dt**3)

        x       = numpy.array([[pos, vel, acc, jerk]]).T

        return x
