import numpy

class AdpativeLQR:

    def __init__(self, n_states, n_inputs, q_noise, r_noise, q_controller, r_controller):

        self.n_states = n_states
        self.n_inputs = n_inputs

        self.q_noise = q_noise
        self.r_noise = r_noise

        self.q_controller = q_controller
        self.r_controller = r_controller

        self.u_prev = numpy.zeros((self.n_inputs, 1))
        self.x_prev = numpy.zeros((self.n_states, 1))

        # intial model
        self.theta = numpy.zeros((self.n_states, self.n_states + self.n_inputs))

        #initial guess
        for n in range(n_states):
            self.theta[n, n] = 1.0

        # intial P covariance matrix
        self.p_system = numpy.eye(self.n_states + self.n_inputs) 

        # forgetting factor
        self.lambda_val = 0.99 

        self.error_sum = numpy.zeros((self.n_states, 1))
    
    def step(self, xr, x):
        #identification
        self.theta, self.p_system = self._identification_step(self.u_prev, self.x_prev, x, self.theta, self.p_system)

        a = self.theta[:, 0:self.n_states]
        b = self.theta[:, self.n_states:]

        #controller synthetis
        self.error_sum+= xr - x
        u, self.p_controller = self._controller_step(a, b, x, self.error_sum, self.p_controller)

        return u
    

    def _controller_step(self, a, b, x, error_sum, p):
        #TODO - add integral action matrix augmentation

        '''
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
        '''

        #riccati DARE solving
        p_new = self.q_controller + a.T @ p @ a - a.T @ p @ b @ numpy.linalg.inv(self.r_controller + b.T @ p @ b) @ b.T @ p @ a

        k_tmp = numpy.linalg.inv(self.r_controller)@(b.T@p_new)

        #split ki for k and integral action part ki
        k   = k_tmp[:, 0:a.shape[0]]
        ki  = k_tmp[:, a.shape[0]:]

        u = - k@x + ki@error_sum

        return u, p_new

    #recursive least squares identification
    def _identification_step(self, u_prev, x_prev, x_now, theta, p):
        # augmented inputs matrix
        extended_x = numpy.concatenate((x_prev, u_prev), axis=0)
        
        # model prediction error
        error = x_now - theta@extended_x

        # Kalman gain    
        #denom = (lambda_val + extended_x.T@P@extended_x).item()
        denom = (self.lambda_val + extended_x.T @ p @ extended_x + x_prev.T @ self.r_noise @ x_prev).item()

        if numpy.abs(denom) > 10e-4 and numpy.abs(denom) < 10e3:
            k = (p@extended_x) / denom
            # model update
            theta_next = theta + (error@k.T)
            # covariance update
            p_next = (1.0 / self.lambda_val) * (p - k@extended_x.T@p) + self.q_noise
        else:
            theta_next = theta
            p_next     = p
        
        return theta_next, p_next
    
