import numpy
import scipy


class KalmanFilter:
    """
        - generic steady state kalman filter

        A : System (state transition) matrix, shape (n_states, n_states)
        B : Control input matrix, shape (n_states, n_inputs)
        H : Measurement matrix, shape (n_measurements, n_states)
        Q : Process noise covariance, shape (n_states, n_states)
        R : Measurement noise covariance, shape (n_measurements, n_measurements)
    """
    def __init__(self, A, B, Q, R, H = None):
     
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        if H is None:
            self.H = numpy.eye(A.shape[0])
        else:
            self.H = H

        # compute steady-state Kalman gain
        self.K = self._solve_kalman_gain(self.A, self.H, self.Q, self.R)

        # initialize state estimate
        self.x_hat = numpy.zeros((A.shape[0], 1))


    """
        y_obs : Measurement vector at the current time step.
        u_in  : Control input vector. if None, assumed zero.
    """
    def step(self, y_obs, u_in):
        # prediction step
        x_hat_new = self.A @ self.x_hat 
        if u_in is not None:
            x_hat_new+= self.B @ u_in

        # correction step
        y_pred = self.H @ x_hat_new
        error  = y_obs.reshape(y_pred.shape) - y_pred

        self.x_hat = x_hat_new + self.K @ error

        return self.x_hat
    
    """
        from curret x_hat, predict future states trajectory
        u_in    : control inputs, of shape (n_steps, n_inputs), optional, set to None of no input
        n_steps : number of prediction steps into future
    """
    def prediction(self, u_in, n_steps):
        x_hat_seq = []  

        x_hat = self.x_hat.copy()

        for n in range(n_steps):
            # prediction step only
            x_hat = self.A @ x_hat
            
            if u_in is not None:
                x_hat+= self.B @ numpy.expand_dims(u_in[n], 1)

            x_hat_seq.append(x_hat[:, 0])

        x_hat_seq = numpy.array(x_hat_seq)

        return x_hat_seq

        
    def _solve_kalman_gain(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        k = p@c.T@scipy.linalg.inv(c@p@c.T + r)

        return k
    

class KalmanFilterPositionVelocityAcceleration:
    def __init__(self, dt, q_process_noise_var, r_measurement_noise_var):
        
        self.dt = dt

        n_states = 3
        
        # constant acceleration model
        mat_a = [
            [1.0, dt, 0.5*(dt**2)],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ]
        
        mat_a = numpy.array(mat_a)

        # system have no input, observer only
        mat_b = numpy.zeros((n_states, 1))

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


        self.kf = KalmanFilter(mat_a, mat_b, mat_q, mat_r)


        # state variables
        self.x2 = 0.0
        self.x1 = 0.0
        self.x0 = 0.0

    

    def step(self, x_obs):
        self.x2 = self.x1
        self.x1 = self.x0
        self.x0 = x_obs

        x = self.x0
        v = (self.x0 - self.x1)/self.dt
        a = (self.x0 - 2*self.x1 + self.x2)/(self.dt**2)

        x_tmp = numpy.array([[x], [v], [a]])

        x_hat = self.kf.step(x_tmp, None)

        return x_hat
    
    def prediction(self, n_steps):
        return self.kf.prediction(None, n_steps)

