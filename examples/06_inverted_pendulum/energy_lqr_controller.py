import numpy
import scipy

class EnergyLQRController:
    def __init__(self, m, l, Ip, Iw, g = 9.81, dt = 0.01):
        # system parameters
        self.m      = m
        self.l      = l
        self.Ip     = Ip
        self.Iw     = Iw
        self.g      = g

     
        self.dt     = dt

        # linearised continuous model
        A = numpy.zeros((4, 4))
        B = numpy.zeros((4, 1))

        a = m*g*l/Ip
        b = 1.0/Ip
        c = 1.0/Iw

        A[0][1] = 1
        A[1][0] = a
        A[2][3] = b
        A[3][0] = -a

        B[1][0] = -b
        B[3][0] = c 

        # swing constant
        self.ke     = 0.2

        # continuous LQR controller
        Q = numpy.zeros((4, 4))
        R = numpy.zeros((1, 1))

        Q[0][0] = 1.0
        R[0][0] = 1.0

        # discrete controller
        self.k = self.solve_lqr(A, B, Q, R)

        print(self.k)

        self.u_max = 1.0

    def step(self, x):
        if abs(x[0]) < 0.5:
            # LQR 
            u =  -self.k @ x
            linear_region = True
        else:
            # swing
            u = self.swing_up_control(x, self.ke)
            linear_region = False

        return numpy.clip(u, -self.u_max, self.u_max), linear_region
    

    def swing_up_control(self, x, kE):
        theta  =  x[0, 0]
        dtheta =  x[1, 0]

        E = 0.5*self.Ip*dtheta**2 + self.m*self.g*self.l*(1 - numpy.cos(theta))

        dE = 0 - E
        u = -kE * dE * numpy.sign(dtheta * numpy.cos(theta) + 1e-6)

        u = u*numpy.ones((1, 1))

        return u


    def solve_lqr(self, a, b, q, r):
        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a, b, q, r)

        # compute the LQR gain
        k = numpy.linalg.inv(r) @ b.T @ p

        return k