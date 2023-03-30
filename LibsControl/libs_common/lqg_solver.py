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

K   : computed controller feedback matrix, NxM
Ki  : computed gain matrix for integral acttion, NxM
F   : kalman gain matrix, NxN
'''
class LQGSolver:

    def __init__(self, a, b, c, q, r, w, dt = 0.001):
        self.a = a
        self.b = b
        self.c = c
        self.q = q
        self.r = r
        self.w = w
        self.dt= dt

    
        
    def solve(self):
        self.k, self.ki = self._find_ki(self.a, self.b, self.c, self.q, self.r)

        self.f = self._find_f(self.a, self.c, self.q, self.w)
        
        return self.k, self.ki, self.f
    
    def closed_loop_response(self, yr, steps = 500, noise = 0.0, disturbance = False):
        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.c.shape[0]  #outputs count

        x       = numpy.zeros((n, 1))


        x_hat   = numpy.zeros((n, 1))
        y       = numpy.zeros((k, 1))
        u       = numpy.zeros((m, 1))

        error_sum = numpy.zeros((k, 1))

        u_result        = numpy.zeros((steps, m))
        x_result        = numpy.zeros((steps, n))
        x_hat_result    = numpy.zeros((steps, n))
        y_result        = numpy.zeros((steps, k))

        for n in range(steps):
            y_obs  = y + noise*numpy.random.randn(k, 1)

            u, x_hat, error_sum = self.forward(u, yr, y_obs, x_hat, error_sum)

            #apply disturbance
            if disturbance == True and n >= steps//2:
                u+= 5 

            #system dynamics step
            x     = x + (self.a@x + self.b@u)*self.dt
            y     = self.c@x

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
    


    def forward(self, u, yr, y, x_hat, error_sum):
        #kalman observer, prediction and correction
        y_hat       = self.c@x_hat
        e           = y - y_hat
        dx_hat      = self.a@x_hat + self.b@u + self.f@e
        x_hat_new   = x_hat + dx_hat*self.dt

        #compute error  
        error     = yr - y 
 
        #integral action
        error_sum_new = error_sum + error*self.dt
        
        #apply controll law
        u = -self.k@x_hat + self.ki@error_sum_new

        return u, x_hat_new, error_sum_new

    
    '''
    solve the continuous time lqr controller.
    dx = A x + B u
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''
    def _find_ki(self, a, b, c, q, r):

        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count
        k = c.shape[0]  #outputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+k, n+k))
        b_aug = numpy.zeros((n+k, m))
        q_aug = numpy.zeros((n+k, n+k))

        a_aug[0:n, 0:n] = a
        a_aug[n:, 0:n]  = c

        b_aug[0:n,0:m]  = b

        #project Q matric to output, and fill augmented q matrix
        tmp = (c@q).sum(axis=1)
        for i in range(k):
            q_aug[n+i][n+i] = tmp[i]


        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        ki_tmp =  scipy.linalg.inv(r)@(b_aug.T@p)

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki
    
    '''
    compute Klaman's gain matrix F
    '''
    def _find_f(self, a, c, q, w):
        p = scipy.linalg.solve_continuous_are(a.T, c.T, q, w)
        f = (p@c.T)@scipy.linalg.inv(w)
       
        return f
    
    