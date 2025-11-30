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
    
