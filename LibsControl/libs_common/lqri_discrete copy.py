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
''' 
class LQRIDiscrete:

    def __init__(self, a, b, q, r, antiwindup = 10**10):
        self.ki, self.k = self.solve(a, b, q, r)

        self.antiwindup = antiwindup


    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, u_prev):
        
        error = xr - x

        du = self.k@error - self.ki@u_prev
        u  = u_prev + du

        #conditional antiwindup
        u = numpy.clip(u, -self.antiwindup, self.antiwindup)
        
        return u


    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r):

        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((m+n, m+n))
        b_aug = numpy.zeros((m+n, m))
        q_aug = numpy.zeros((m+n, m+n))

        #add integrator into augmented A matrix
        for i in range(m):
            a_aug[i, i] = 1.0

        #place A matrix in
        a_aug[m:, m:] = a

        #place B matrix in
        a_aug[m:, 0] = b[:, 0]


        #add integrator into augmented B matrix
        for i in range(m):
            b_aug[i, i] = 1.0
           
        #project Q matric to output, and fill augmented q matrix
        q_aug[m:, m:] = q

        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        k_tmp = numpy.linalg.inv(r)@(b_aug.T@p)

        #truncate small elements (due numerical errors)
        #k[numpy.abs(k) < 10**-10] = 0

     
        #split ki for k and integral action part ki
        ki  = k_tmp[:, 0:m]
        k   = k_tmp[:, m:]

        print(k_tmp)


        return ki, k