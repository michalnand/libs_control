import numpy
import scipy

class GainSchedulingLQRController:
    def __init__(self, m, l, Ip, Iw, g = 9.81, dt = 0.01):
        # system parameters
        self.a = m * g * l / Ip
        self.b = 1.0 / Ip
        self.c = 1.0 / Iw

     
        self.dt     = dt

        # continuous LQR controller
        Q = numpy.diag([1.0, 1e-7, 1e-7, 1e-7])
        R = numpy.ones((1, 1))

        self.k_zero = self._lqr_gain(Q, R, linearization="upright")
        print(self.k_zero)

        Q = numpy.diag([1.0, 1e-7, 1e-7, 1e-7])
        R = numpy.ones((1, 1))

        self.k_pi = self._lqr_gain(Q, R, linearization="downward")
        print(self.k_pi)

        self.u_max = 1.0

    def step(self, xr, x):
        theta = x[0, 0]

        alpha = self._gain_schedule(theta)
        K = (1 - alpha) * self.k_zero + alpha * self.k_pi

        u = K @ (xr - x)
        u = numpy.clip(u, -self.u_max, self.u_max)

        return u, alpha



    def _lqr_gain(self, Q, R, linearization="upright"):
        A = numpy.zeros((4, 4))
        B = numpy.zeros((4, 1))

        if linearization == "upright":
            a =  self.a
        elif linearization == "downward":
            a = -self.a
        else:
            raise ValueError("Unknown linearization")

        A[0, 1] = 1
        A[1, 0] = a
        A[2, 3] = self.b
        A[3, 0] = -a

        B[1, 0] = -self.b
        B[3, 0] =  self.c


        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = numpy.linalg.inv(R) @ B.T @ P

        return K
    
    def _gain_schedule(self, theta):
        """
        Smooth scheduling variable: 0 at 0 rad, 1 at ±pi
        """
        return 0.5 * (1 - numpy.cos(theta))
