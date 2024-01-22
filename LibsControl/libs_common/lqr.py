import numpy
import scipy


'''
solve LQR controller for continuous dynamical system
dx = Ax + Bu

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)
''' 
class LQR:

    def __init__(self, a, b, q, r):
        self.k, self.ki = self.solve(a, b, q, r)

    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, integral_action, dt):

        #integral action
        error = xr - x
        integral_action_new = integral_action + self.ki@error*dt

        #LQR controll law
        u = -self.k@x + integral_action_new

        return u, integral_action_new

    
    '''
    solve the continuous time lqr controller.
    dx = A x + B u
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
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
        a_aug[n:, 0:n]  = numpy.eye(n)

        b_aug[0:n,0:m]  = b

        #project Q matric to output, and fill augmented q matrix
        q_aug[n:, n:] = q

        # continuous-time algebraic Riccati equation solution
        p = scipy.linalg.solve_continuous_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        ki_tmp =  numpy.linalg.inv(r)@(b_aug.T@p)

        #truncated small elements
        ki_tmp[numpy.abs(ki_tmp) < 10**-10] = 0

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki

