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


    def __init__(self, a, b, c, q, r, w, dt = 0.001):
        self.a = a
        self.b = b
        self.c = c
        self.q = q
        self.r = r
        self.w = w
        self.dt= dt

      
    def solve(self): 
        self.k, self.ki, self.g = self._find_kig(self.a, self.b, self.c, self.q, self.r)

        self.f  = self._find_f(self.a, self.c, self.q, self.w)

        return self.k,  self.ki, self.g, self.f
    
    def closed_loop_response(self, yr, steps = 500, observation_noise = 0.0, disturbance = False):

        x       = numpy.zeros((self.a.shape[0], 1))
        
        x_hat   = numpy.zeros((self.a.shape[0], 1))
        y_hat   = numpy.zeros((self.c.shape[0], 1))
        y       = numpy.zeros((self.c.shape[0], 1))

        u       = numpy.zeros((self.b.shape[1], 1))

        u_result        = numpy.zeros((steps, self.b.shape[1]))
        x_result        = numpy.zeros((steps, self.a.shape[0]))
        x_hat_result    = numpy.zeros((steps, self.a.shape[0]))
        y_result        = numpy.zeros((steps, self.c.shape[0]))
        
        #error integral
        error_int = numpy.zeros((self.c.shape[0], 1))

        for n in range(steps):
            #kalman observer
            y_hat   = self.c@x_hat
            e       = y - y_hat
            dx_hat  = self.a@x_hat + self.b@u + self.f@e
            x_hat   = x_hat + dx_hat*self.dt
            
            #integral action
            #TODO : antiwindup
            error = yr*self.g - y
            error_int+= error*self.dt

            #apply LQR control law
            u = self.ki@error_int - self.k@x_hat

            #disturbance for testing
            if disturbance == True and n > steps//2:
                u+= 10.0
            
            #system dynamics step
            x       = x + (self.a@x + self.b@u)*self.dt
            y       = self.c@x
            
            y       = y + observation_noise*numpy.random.randn(y.shape[0], y.shape[1])

           
            u_result[n]     = u[:, 0]
            x_result[n]     = x[:, 0]
            x_hat_result[n] = x_hat[:, 0]
            y_result[n]     = y[:, 0]

        return u_result, x_result, x_hat_result, y_result

     
    def get_poles(self):
        
        poles_ol = numpy.linalg.eigvals(self.a) + 0j
        re_ol = poles_ol.real
        im_ol = poles_ol.imag

        poles_cl = numpy.linalg.eigvals(self.a - self.b@self.k) + 0j
        re_cl = poles_cl.real
        im_cl = poles_cl.imag

        return re_ol, im_ol, re_cl, im_cl
    
    '''
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

    '''
    solve the continuous time lqr controller.
    dx = A x + B u
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''
    def _find_kig(self, a, b, c, q, r):
        
        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.c.shape[0]  #outputs count

        a_aug  = numpy.zeros((n+k, n+k))
        b_aug  = numpy.zeros((n+k, m))
        q_aug  = numpy.zeros((n+k, n+k))
    
        a_aug[0:n, 0:n] = a
        a_aug[n:, 0:n]  = c
        b_aug[0:n, 0:m] = b

        q_aug[0:n, 0:n] = q

        
        numpy.fill_diagonal(q_aug[n:, n:], 10.0)

       
        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a_aug, b_aug, q_aug, r)
        
        # compute the LQR gain
        ki_tmp =  scipy.linalg.inv(r) @ (b_aug.T@p)

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        #find scaling for reference value using steady state response
        x_steady_state = -numpy.linalg.pinv(a_aug - b_aug@ki_tmp)@b_aug@ki_tmp
        g = 1.0/numpy.diagonal(x_steady_state)
        g = numpy.expand_dims(g, 1)
        
        g = g[n:]
        
        return k, ki, g
    
  

    def _find_f(self, a, c, q, w):
        
        # continuous-time algebraic Riccati equation solution


        p = scipy.linalg.solve_continuous_are(a.T, c.T, q, w)
        

        f = (p@c.T)@scipy.linalg.inv(w)
       
        return f
    