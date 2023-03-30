import numpy
import scipy.linalg
import matplotlib.pyplot as plt


'''
solve LQR for dynamical system : 
dx = A@x + B@u
y  = x

controller : 
u = K@(Gx_r - x) 

where : 
N : system order
M : system inputs count

A : system dynamic matrix, NxN
B : system input   matrix, NxM
C : diagonal matrix, NxN, projecting full state x to output y

x   : state column vector, Nx1
x_r : required state, same shape as x
u   : control input, column vector, Mx1

Q : diagonal matrix, NxN, weighting for required state elements
R : diagonal amtrix, MxM, weighting for controll value

returns : 

K : computed controller feedback matrix, NxM
G : computed required state (x_r) scaling matrix, to remove steady state error, Nx1
'''
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
    
    def closed_loop_response(self, xr, steps = 500, noise = 0.0, disturbance = False):
        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.c.shape[0]  #outputs count
        
        x  = numpy.zeros((n, 1))
 
        u_result = numpy.zeros((steps, m))
        x_result = numpy.zeros((steps, n))
        y_result = numpy.zeros((steps, k))


        for n in range(steps):
            x_obs       = x + noise*numpy.random.randn(x.shape[0], x.shape[1])

            u = self.forward(xr, x_obs)

            #apply disturbance
            if disturbance == True and n >= steps//2:
                u+= 1

            #system dynamics step
            x     = x + (self.a@x + self.b@u)*self.dt
            y     = self.c@x

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]
            y_result[n] = y[:, 0]

        return u_result, x_result, y_result
    
    def get_poles(self):
        
        poles_ol = numpy.linalg.eigvals(self.a) + 0j
        re_ol = poles_ol.real
        im_ol = poles_ol.imag

        poles_cl = numpy.linalg.eigvals(self.a - self.b@self.k) + 0j
        re_cl = poles_cl.real
        im_cl = poles_cl.imag

        return re_ol, im_ol, re_cl, im_cl
    

    def forward(self, xr, x):
        #compute error
        error = xr*self.g - x

        #apply controll law
        u = self.k@error
        return u

    
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
        k =  scipy.linalg.inv(r)@(b.T@p)
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
   