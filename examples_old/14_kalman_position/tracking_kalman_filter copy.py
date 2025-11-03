import numpy
import scipy


class TrackingKalmanFilter:
    """
       
    """
    def __init__(self,  q_process_noise_var, r_measurement_noise_var, n_dims, n_objects, dt):
        self.n_dims      = n_dims
        self.n_objects   = n_objects
        self.dt          = dt
        
        
        self.A, self.R, self.Q = self._get_matrices(q_process_noise_var, r_measurement_noise_var, n_dims, dt)
        
        self.H = numpy.eye(self.A.shape[0])  

        # compute steady-state Kalman gain
        self.K = self._solve_kalman_gain(self.A, self.H, self.Q, self.R)

        # initialize state estimate
        self.x_hat = numpy.zeros((n_objects, self.A.shape[0]))

        self.x0 = numpy.zeros((n_objects, self.n_dims))
        self.x1 = numpy.zeros((n_objects, self.n_dims))
        self.x2 = numpy.zeros((n_objects, self.n_dims))

    """
        x : meassured 2D position, shape (n_objects, n_dims)
        returns filtered positions
    """
    def step(self, x_obs):
        self.x2 = self.x1.copy()
        self.x1 = self.x0.copy()
        self.x0 = x_obs.copy()

        # differentiate to obtain velocity and acceleration
        x = self.x0
        v = (self.x0 - self.x1)/self.dt
        a = (self.x0 - 2*self.x1 + self.x2)/(self.dt**2)    

        #x_tmp = numpy.concatenate([x, v, a], axis=1)
        x_tmp = numpy.zeros((self.n_objects, self.n_dims*3))
        for i in range(self.n_dims):    
            x_tmp[:, 3*i + 0] = x[:, i]
            x_tmp[:, 3*i + 1] = v[:, i]
            x_tmp[:, 3*i + 2] = a[:, i]

        # prediction step
        x_hat_new =  self.x_hat@self.A.T
     
        # correction step
        error  = x_tmp - x_hat_new  
        self.x_hat = x_hat_new + error@self.K.T

        return self.x_hat[:, 0:self.n_dims]
    

    def prediction(self, n_steps):
        x_hat_seq = []  

        x_hat = self.x_hat.copy()

        indices = []
        for n in range(self.n_dims):
            indices.append(n*3) 

        for n in range(n_steps):
            x_hat = x_hat@self.A.T  

            x_hat_seq.append(x_hat[:, indices])

        x_hat_seq = numpy.array(x_hat_seq)

        
        return x_hat_seq
    
    '''
        set state - position, for object_idx
        if x_initial is None, fill with zeros
        x_initial.shape = (n_dims, )
    '''
    def set_state(self, object_idx, x_initial = None):
        self.x_hat[object_idx, :] = 0

        if x_initial is not None:
            indices = []
            for n in range(self.n_dims):
                indices.append(n*3) 

            n = len(x_initial.shape)
            self.x_hat[object_idx, indices] = x_initial


    
        
    def _solve_kalman_gain(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        k = p@c.T@scipy.linalg.inv(c@p@c.T + r)

        return k
    

    def _get_matrices(self, q_process_noise_var, r_measurement_noise_var, n_dims, dt):
        n_states = 3
        
        # constant acceleration model
        mat_a = [
            [1.0, dt, 0.5*(dt**2)],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ]
        
        mat_a = numpy.array(mat_a)


        # measurement noise estimation from position noise only
        mat_r = [
            [1.0, 1.0/dt, 1.0/(dt**2)],
            [1.0/dt, 2.0/(dt**2), 3.0/(dt**3)],
            [1.0/(dt**2), 3.0/(dt**3), 6.0/(dt**4)]
        ]

        mat_r = numpy.array(mat_r)
        mat_r = r_measurement_noise_var*mat_r

      
        # process noise estimation from position noise only
        mat_q = [
            [(dt**5)/20.0, (dt**4)/8.0, (dt**3)/6.0],
            [(dt**4)/8.0, (dt**3)/3.0, (dt**2)/2.0],
            [(dt**3)/6.0, (dt**2)/2.0, dt]
        ]

        mat_q = numpy.array(mat_q)
        mat_q = q_process_noise_var*mat_q


        # stack matrices along diagonal, for n_dims motion
        mat_a_res = numpy.zeros((n_states*n_dims, n_states*n_dims))
        for n in range(n_dims):
            mat_a_res[n*n_states:(n+1)*n_states, n*n_states:(n+1)*n_states] = mat_a

        mat_r_res = numpy.zeros((n_states*n_dims, n_states*n_dims))
        for n in range(n_dims):
            mat_r_res[n*n_states:(n+1)*n_states, n*n_states:(n+1)*n_states] = mat_r

        mat_q_res = numpy.zeros((n_states*n_dims, n_states*n_dims))
        for n in range(n_dims):
            mat_q_res[n*n_states:(n+1)*n_states, n*n_states:(n+1)*n_states] = mat_q

        return mat_a_res, mat_r_res, mat_q_res
        
        