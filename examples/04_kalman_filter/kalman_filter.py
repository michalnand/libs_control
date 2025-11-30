import numpy
import scipy


class KalmanFilter:
    """
        - generic steady state kalman filter

        A : System (state transition) matrix, shape (n_states, n_states)
        H : Measurement matrix, shape (n_measurements, n_states)
        Q : Process noise covariance, shape (n_states, n_states)
        R : Measurement noise covariance, shape (n_measurements, n_measurements)
    """
    def __init__(self, A, Q, R, H = None):
     
        self.A = A
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
    def step(self, y_obs):
        # prediction step
        x_hat_new = self.A @ self.x_hat 
      
        # correction step
        y_pred = self.H @ x_hat_new
        error  = y_obs.reshape(y_pred.shape) - y_pred

        self.x_hat = x_hat_new + self.K @ error

        return self.x_hat
    
   

        
    def _solve_kalman_gain(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        k = p@c.T@scipy.linalg.inv(c@p@c.T + r)

        return k
    
