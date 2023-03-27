import numpy
import scipy.linalg
import matplotlib.pyplot as plt


'''
solve LQR for dynamical system : 
dx = A@x + B@u

controller : 
u = -K@x + Gx_r

where : 
N : system order
M : system inputs count

A : system dynamic matrix, NxN
B : system input   matrix, NxM

x   : state column vector, Nx1
x_r : required state, same shape as x
u   : control input, column vector, Mx1

Q : diagonal matrix, NxN, weighting for required state elements
R : diagonal amtrix, MxM, weighting for controll value

returns : 

K : computed controller feedback matrix, NxM
G : computed required state (x_r) scaling matrix, to remove steady state error, Nx1
'''
class LQRISolver:

    def __init__(self, a, b, c, q, r, dt):

        self.a = a
        self.b = b
        self.c = c

        self.q = q
        self.r = r

        self.dt= dt
        
    def solve(self):
        self.k, self.ki = self._find_ki(self.a, self.b, self.c, self.q, self.r)
        
        return self.k, self.ki
    
    def closed_loop_response(self, xr, steps = 500, noise = 0.0, disturbance = False):
        u_result, x_result, y_result = self._closed_loop_response(self.a, self.b, self.c, xr, self.k, self.ki, steps, noise, disturbance)

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
    def _find_ki(self, a, b, c, q, r):

        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.c.shape[0]  #outputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+k, n+k))
        b_aug = numpy.zeros((n+k, m))
        q_aug = numpy.zeros((n+k, n+k))

        a_aug[0:n, 0:n] = a
        a_aug[n:, 0:n]  = c

        b_aug[0:n,0:m]  = b

        q_aug[0:n,0:n]  = 0
        q_aug[k:,k:]    = q

        
        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        ki_tmp =  scipy.linalg.inv(r)@(b_aug.T@p)

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki
    
    '''
    find scaling for reference value using steaduy state response
    '''
    def _find_g(self, a, b, k):
        x_steady_state = -numpy.linalg.pinv(a-b@k)@b@k
        #y_steady_state = c@x_steady_state
        g = 1.0/numpy.diagonal(x_steady_state)
        g = numpy.expand_dims(g, 1)

        return g
    
    def _closed_loop_response(self, a, b, c, xr, k, ki, steps = 500, noise = 0.0, disturbance = False):

        x  = numpy.zeros((a.shape[0], 1))
 
        u_result = numpy.zeros((steps, b.shape[1]))
        x_result = numpy.zeros((steps, a.shape[0]))
        y_result = numpy.zeros((steps, c.shape[0]))

        u = numpy.zeros((1, c.shape[0]))

        error_sum = numpy.zeros((1, b.shape[1]))

        for n in range(steps):
            x_obs       = x + noise*numpy.random.randn(x.shape[0], x.shape[1])

            #compute error
            error     = xr - x_obs

            #integral action
            error_sum = error_sum + error*self.dt

            #apply controll law
            u = -k@x_obs + ki@error_sum

          
            #apply disturbance
            if disturbance == True and n >= steps//2:
                u+= 5

            #system dynamics step
            x     = x + (a@x + b@u)*self.dt
            y     = c@x

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]
            y_result[n] = y[:, 0]

        return u_result, x_result, y_result
            
