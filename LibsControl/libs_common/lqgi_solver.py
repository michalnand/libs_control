import numpy
import scipy.linalg
import matplotlib.pyplot as plt

'''
solve LQG for dynamical system : 
dx = A@x + B@u
y  = Cx

controller : 
u = -K@x_hat + Gx_r

Kalman observer : 
dx_hat = Ax_hat + Bu + F(y - y_hat)
y_hat  = Cx_hat

where : 
N : system order
M : system inputs count
K : system outputs count

A : system dynamic matrix, NxN
B : system input   matrix, NxM
C : system output matrix,  KxM

x   : state column vector, Nx1
x_r : required state, same shape as x
u   : control input, column vector, Mx1

Q : diagonal matrix, NxN, weighting for required state elements
R : diagonal matrix, MxM, weighting for controll value
V : observation noise variance matrix, KxK
W : process noise variance matrix, NxN, (optional)


returns : 

K : computed controller feedback matrix, NxM
G : computed required state (x_r) scaling matrix, to remove steady state error, Nx1
F : kalman gain matrix, NxN
'''

class LQGISolver:


    def __init__(self, a, b, c, q, r, w = None, dt = 0.001):
        self.a = a
        self.b = b
        self.c = c
        self.q = q
        self.r = r
        self.w = w
        self.dt= dt

        if self.w is None:
            self.w = numpy.zeros(self.c.shape)


        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.c.shape[0]  #outputs count

        self.a_aug  = numpy.zeros((m+n, m+n))
        self.b_aug  = numpy.zeros((m+n, m))
        self.c_aug  = numpy.zeros((k, m+n)) 
        self.q_aug  = numpy.zeros((m+n, m+n))
        

        self.a_aug[m:, 0:m]     = self.b
        self.a_aug[m:,  m:]      = self.a

        numpy.fill_diagonal(self.b_aug[0:m, 0:m], 1.0)

        self.c_aug[:,  m:]        = self.c
        self.q_aug[m:, m:]       = self.q

        numpy.fill_diagonal(self.q_aug[0:m, 0:m], 0.1)
 
        
    def solve(self):
        self.k  = self._find_k(self.a_aug, self.b_aug, self.q_aug, self.r)

        self.g  = self._find_g(self.a_aug, self.b_aug, self.c_aug, self.k)

        self.f  = self._find_f(self.a, self.c, self.q, self.w)

        return self.k, self.g, self.f
    
    def closed_loop_response(self, xr, steps = 500, observation_noise = 0.0, disturbance = False):

        x       = numpy.zeros((self.a.shape[0], 1))
        
        x_hat   = numpy.zeros((self.a.shape[0], 1))
        y_hat   = numpy.zeros((self.c.shape[0], 1))
        y       = numpy.zeros((self.c.shape[0], 1))

        u       = numpy.zeros((self.b.shape[1], 1))

        u_result        = numpy.zeros((steps, self.b.shape[1]))
        x_result        = numpy.zeros((steps, self.a.shape[0]))
        x_hat_result    = numpy.zeros((steps, self.a.shape[0]))
        y_result        = numpy.zeros((steps, self.c.shape[0]))
        
        xru = numpy.zeros(self.g.shape)
        xru[u.shape[0]:,:] = xr


        for n in range(steps):
            #kalman observer
            y_hat   = self.c@x_hat
            e       = y - y_hat
            dx_hat  = self.a@x_hat + self.b@u + self.f@e
            x_hat   = x_hat + dx_hat*self.dt
            
            #apply LQR control law

            xu = numpy.vstack([u, x_hat])

            error   = xru*self.g - xu
            #error   = -xu

            du       = self.k@error

            u = u + du*self.dt
                        
            
            #system dynamics step
            x       = x + (self.a@x + self.b@u)*self.dt
            y       = self.c@x
            
            y       = y + observation_noise*numpy.random.randn(y.shape[0], y.shape[1])

           
            #disturbance for testing
            if disturbance == True and n == steps//2:
                x+= numpy.abs(x)
                
            
            u_result[n]     = u[:, 0]
            x_result[n]     = x[:, 0]
            x_hat_result[n] = x_hat[:, 0]
            y_result[n]     = y[:, 0]

        return u_result, x_result, x_hat_result, y_result

     
    def get_poles(self):
        
        poles_ol = numpy.linalg.eigvals(self.a) + 0j
        re_ol = poles_ol.real
        im_ol = poles_ol.imag

        poles_cl = numpy.linalg.eigvals(self.a_aug - self.b_aug@self.k) + 0j
        re_cl = poles_cl.real
        im_cl = poles_cl.imag

        return re_ol, im_ol, re_cl, im_cl
    
    def forward(self, x_hat, u, xr, y):
        #kalman observer
        y_hat   = self.c@x_hat
        e       = y - y_hat
        dx_hat  = self.a@x_hat + self.b@u + self.f@e
        x_hat   = x_hat + dx_hat*self.dt
        
        #apply LQR control law
        error   = xr*self.g - x_hat
        u       = self.k@error

        return x_hat, u

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
        k =  scipy.linalg.inv(r) @ (b.T@p)
        return k
    
    '''
    find scaling for reference value using steaduy state response
    '''
    def _find_g(self, a, b, c, k):

        x_steady_state = -numpy.linalg.pinv(a - b@k)@b@k
        #y_steady_state = x_steady_state
        g = 1.0/numpy.diagonal(x_steady_state)
        g = numpy.expand_dims(g, 1)

        return g
    

    def _find_f(self, a, c, q, w):
        
        # continuous-time algebraic Riccati equation solution


        p = scipy.linalg.solve_continuous_are(a.T, c.T, q, w)
        

        f = (p@c.T)@scipy.linalg.inv(w)
       
        return f
    