import numpy

class AdpativeMPC:

    def __init__(self, n_states, n_inputs, q):

        self.n_states = n_states
        self.n_inputs = n_inputs

        self.q = q

        self.u_prev = numpy.zeros((self.n_inputs, 1))
        self.x_prev = numpy.zeros((self.n_states, 1))

        # intial model
        self.theta = numpy.zeros((self.n_states, self.n_states + self.n_inputs))

        # intial P covariance matrix
        self.p_system = numpy.eye(self.n_states + self.n_inputs) 

        # forgetting factor
        self.lambda_val = 0.99 

    
    def step(self, xr, x):
        #identification
        self.theta_next, self.p_system = self._identification_step(self.u_prev, self.x_prev, x, self.theta, self.p_system)

        a = self.theta_next[:, 0:self.n_states]
        b = self.theta_next[:, self.n_states:]

        #TODO : do we need matrix a, b, q augmentations for integral action ?
        #compute optimal control for given u
        q_inv = numpy.inv(self.q)
        u = numpy.linalg.inv(b) @ numpy.linalg.inv(a.T@q_inv@a) @ (a@x - xr)

        self.u_prev = u.copy() 
        self.x_prev = x.copy()

        return u

    #recursive least squares identification
    def _identification_step(self, u_prev, x_prev, x_now, theta, p):
        
        # augmented inputs matrix
        extended_x = numpy.concatenate((x_prev, u_prev), axis=0)
        
        # model prediction error
        error = x_now - self.theta@extended_x

        # Kalman gain    
        denom = (self.lambda_val + extended_x.T@p@extended_x)[0][0]
        if numpy.abs(denom) > 10e-4 and numpy.abs(denom) < 10e3:
            k = (p@extended_x) / denom
            # model update
            theta_next = theta + (error@k.T)
            # covariance update
            p_next = (1.0 / self.lambda_val) * (p - k@extended_x.T@p)
        else:
            theta_next = theta
            p_next     = p

        return theta_next, p_next

