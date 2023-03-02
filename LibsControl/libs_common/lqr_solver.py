import numpy
import scipy.linalg
import matplotlib.pyplot as plt



class LQRSolver:

    def __init__(self, a, b, c, q, r, dt):

        self.a = a
        self.b = b
        self.c = c

        self.q = q
        self.r = r

        self.dt= dt
        
    def solve(self):
        self.k = self._find_k(self.a, self.b, self.q, self.r)
        self.g = self._find_g(self.a, self.b, self.k)

        return self.k, self.g
    
    def closed_loop_response(self, xr, steps = 500):
        u_result, x_result, y_result = self._closed_loop_response(self.a, self.b, self.c, xr, self.k, self.g, steps)

        return u_result, x_result, y_result
     
    def get_poles(self):
        
        poles_ol = numpy.linalg.eigvals(self.a) + 0j
        re_ol = poles_ol.real
        im_ol = poles_ol.imag

        poles_cl = numpy.linalg.eigvals(self.a - self.b@self.k) + 0j
        re_cl = poles_cl.real
        im_cl = poles_cl.imag

        return re_ol, im_ol, re_cl, im_cl
    
    '''
    solve the continuous time lqr controller.
    dx = A x + B u
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''
    def _find_k(self, a, b, q, r):
       
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
    
    '''
    find scaling for reference value using steaduy state response
    '''
    def _find_g(self, a, b, k):
        x_steady_state = -numpy.linalg.pinv(a-b@k)@b@k
        #y_steady_state = c@x_steady_state
        g = 1.0/numpy.diagonal(x_steady_state)
        g = numpy.expand_dims(g, 1)

        return g
    
    def _closed_loop_response(self, a, b, c, xr, k, g = None, steps = 500):

        x  = numpy.zeros((a.shape[0], 1))

        
        u_result = numpy.zeros((steps, b.shape[1]))
        x_result = numpy.zeros((steps, a.shape[0]))
        y_result = numpy.zeros((steps, c.shape[0]))

        if g is None:
            g = numpy.ones_like(xr)

        for n in range(steps):
            x_obs       = x + 0.1*numpy.random.randn(x.shape[0], x.shape[1])

            #compute error, include gain scaling matrix
            error = xr*g - x_obs

            #apply controll law
            u = k@error

        
            #system dynamics step
            x     = x + (a@x + b@u)*self.dt

            
            if n == steps//2:
                x[1, 0]+= 1


            y     = c@x

            
            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]
            y_result[n] = y[:, 0]

        return u_result, x_result, y_result
            
