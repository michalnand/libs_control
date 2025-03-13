import numpy
import scipy


'''
solve LQR controller for discrete discrete system
x(n+1) = Ax(n) + Bu(n)

Q, R are weight matrices in quadratic loss

R delta is matrix penalting u change

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q           matrix, shape (n_states, n_states)
R           matrix, shape (n_inputs, n_inputs)

control law : 
e(n)    = x(n) - xr(n)
z(n)    = [e(n), u(n)]
du(n)   = -K*z(n)
u(n+1)  = u(n) + du(n)
'''  
class LQRIDiscrete:

    def __init__(self, a, b, q, r, antiwindup = 10**10):
        self.k = self.solve(a, b, q, r)

        self.antiwindup = antiwindup

        self.v = numpy.zeros((a.shape[1], 1))


    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)

    returns:
        u_curr : input into plant, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, u_curr):

        # Compute tracking error
        #error = x - xr  # (n x 1)

        error = xr - x  # (n x 1)
        
        # Form the augmented state [error; previous control]
        z = numpy.vstack([error, u_curr])  # (n+m x 1)
        
        # Compute control increment (delta u)
        du = self.k @ z   # (m x 1)
        
        # Update stored control
        u_resp = u_curr + du
        
        # Apply anti-windup clipping if desired
        u_resp = numpy.clip(u_resp, -self.antiwindup, self.antiwindup)

        return u_resp



    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r):
    
        n = a.shape[0]   # state dimension
        m = b.shape[1]   # input dimension

        # Correct augmented system: note the top-right block is B, not zeros.
        a_aug = numpy.block([
            [a, b],
            [numpy.zeros((m, n)), numpy.eye(m)]
        ])

        print(a_aug)

        b_aug = numpy.block([
            [numpy.zeros((n, m))],
            [numpy.eye(m)],
        ])

        print(b_aug)

        # Define the augmented cost weighting matrix.
        # Here we penalize the error (using Q) and the control increment (using R).
        # Optionally, you can add a light weight on u itself in the lower-right block.
        q_aug = numpy.block([
            [q,               numpy.zeros((n, m))],
            [numpy.zeros((m, n)),  numpy.zeros((m, m))]
        ])

        # Solve the discrete-time algebraic Riccati equation (DARE)
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # Compute the LQR gain for the augmented system
        k = numpy.linalg.inv(r + b_aug.T @ p @ b_aug) @ (b_aug.T @ p @ a_aug)

        return k
        
