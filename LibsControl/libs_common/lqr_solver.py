import numpy
import scipy.linalg
import matplotlib.pyplot as plt



class LQRSolver:

    def __init__(self, a, b, q, r, dt):
        self.a = a
        self.b = b
        self.q = q
        self.r = r
        self.dt= dt
        
    def solve(self):
        self.k = self._find_k(self.a, self.b, self.q, self.r)
        self.g = self._find_g(self.a, self.b, self.q, self.k)

        return self.k, self.g
    
    def closed_loop_response(self, xr, steps = 500, decimation=1):
        u_result, x_result = self._closed_loop_response(self.a, self.b, xr, self.k, self.g, steps, decimation)

        return u_result, x_result
     
    def get_poles(self):
        
        poles_ol = numpy.linalg.eigvals(self.a) + 0j
        re_ol = poles_ol.real
        im_ol = poles_ol.imag

        poles_cl = numpy.linalg.eigvals(self.a - self.b@self.k) + 0j
        re_cl = poles_cl.real
        im_cl = poles_cl.imag

        return re_ol, im_ol, re_cl, im_cl

    def _find_k(self, a, b, q, r):
        """
        solve the continuous time lqr controller.
        dx = A x + B u
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a, b, q, r)

        '''
        r_inv = scipy.linalg.inv(r)
        p = numpy.zeros_like(a)
        dt = 0.01
        for i in range(1000):
            dp = a.T@p + p@a - (p@b)@r_inv@(b.T@p) + q
            p+= dp*dt
        '''
        
        # compute the LQR gain
        k =  scipy.linalg.inv(r) * (b.T@p)
        return k
    
    def _find_g(self, a, b, q, k, step_size = 0.1):
        order = a.shape[0]
        g = numpy.ones((order, 1))

        #TODO : compute gains g from step response
        for i in range(order):
            xr      = numpy.zeros((order, 1))
            xr[i]   = step_size 

            u_result, x_result = self._closed_loop_response(a, b, xr, k, None, 8000) 

            x_res   = x_result[-1][i]

            if numpy.abs(q[i][i]) > 0:
                g[i]    = step_size/x_res
            else:
                g[i]    = 0.0

        g = numpy.clip(g, -1000, 1000)

        return g
    
    def _closed_loop_response(self, a, b, xr, k, g = None, steps = 500, decimation=1):

        x  = numpy.zeros((a.shape[0], 1))

        x_result = numpy.zeros((steps, a.shape[0]))
        u_result = numpy.zeros((steps, b.shape[1]))


        if g is None:
            g = numpy.ones_like(xr)

        for n in range(steps):
            #compute error, include gain scaling matrix
            error = xr*g - x


            #apply controll law
            if n%decimation == 0:
                u     = k@error
            
            #system dynamics step
            x     = x + (a@x + b@u)*self.dt

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]

        return u_result, x_result
            
