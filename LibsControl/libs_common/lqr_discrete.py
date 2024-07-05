import numpy
import scipy


'''
solve LQR controller for discrete discrete system
x(n+1) = Ax(n) + Bu(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)

control law : 
e_sum(n)= e_sum(n-1) + xr(n) - x(n)
u(n)    = -K*x(n) + Ki*e_sum(n)

'''  
class LQRDiscrete:

    def __init__(self, a, b, q, r, antiwindup = 10**10, di_max = 10**10):
        self.k, self.ki = self.solve(a, b, q, r)

        self.antiwindup = antiwindup
        self.di_max     = di_max


    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, integral_action):
        #integral action
        error = xr - x

        dintegral_action    = self.ki@error

        #kick clipping
        dintegral_action    = numpy.clip(dintegral_action, -self.di_max , self.di_max)
        integral_action_new = integral_action + dintegral_action

        #LQR controll law
        u_new = -self.k@x + integral_action

        #conditional antiwindup
        u = numpy.clip(u_new, -self.antiwindup, self.antiwindup)
        integral_action_new = integral_action_new - (u_new - u)

        return u, integral_action_new


    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r):

        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count
        k = a.shape[0]  #outputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+k, n+k))
        b_aug = numpy.zeros((n+k, m))
        q_aug = numpy.zeros((n+k, n+k))

        
        a_aug[0:n, 0:n] = a 

        #add integrator into augmented a matrix
        for i in range(n):
            a_aug[i + n, i]     = 1.0
            a_aug[i + n, i + n] = 1.0

        b_aug[0:n,0:m]  = b

        #project Q matric to output, and fill augmented q matrix
        q_aug[n:, n:] = q

        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        ki_tmp =  numpy.linalg.inv(r)@(b_aug.T@p)

        #truncated small elements
        ki_tmp[numpy.abs(ki_tmp) < 10**-10] = 0

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki