import numpy
import scipy

from .optim_adam import OptimAdam
from .optim_sgd  import OptimSGD

"""
    gradient based model predictive control MPC

    system to control :
        x(n+1) = Ax(n) + Bu(n)
        u(n+1) = u(n) + du(n)

    constrains : 
        u  = clip(u,  u_min, u_max)     : actuator saturation
        du = clip(du, du_min, du_max)   : maximum change constrain - kick avoid
"""

class GradientMPC:

    def __init__(self, mat_a, mat_b, mat_q, mat_r, prediction_horizon = 16, control_horizon = 4, Optimizer = None):
        self.n_states = mat_a.shape[0]
        self.n_inputs = mat_b.shape[1]  


        self.phi, self.omega, self.theta, self.q_aug, self.r_aug = self._matrix_augmentation(mat_a, mat_b, mat_q, mat_r, control_horizon, prediction_horizon)

        self.Optimizer = Optimizer

       
        self.u_min = -1.0
        self.u_max =  1.0
        self.du_min = -1.0
        self.du_max = 1.0



        self.du = None
   
    
    def forward(self, xr, x, u_prev, n_iterations = 8, learning_rate = 0.1, constrain_func = None):  
        xr_aug = numpy.reshape(xr, (xr.shape[0] * xr.shape[1], 1))

        # Initialize control increments
        du = numpy.zeros((self.theta.shape[1], 1))

        # use old solution as initial
        if self.du is not None:
            du = self.du.copy()

        optimizer = self.Optimizer(du, learning_rate)

      

        # Optimization loop
        for _ in range(n_iterations):
            # Predict the trajectory with current du
            x_hat = self.phi @ x + self.omega @ u_prev + self.theta @ du

            # Compute gradient of cost with respect to du
            error = xr_aug - x_hat
            grad = 2 * (self.theta.T @ self.q_aug @ error) + 2 * self.r_aug @ du

            # Update du with gradient descent step
            #du = -learning_rate * grad
            du = optimizer.step(du, grad)

            # Apply clipping to du
            du = numpy.clip(du, self.du_min, self.du_max)

        self.du = du.copy()


        # Compute final control output u
        if constrain_func is not None:
            u = constrain_func(u_prev + du[:self.n_inputs])
        else:
            u = numpy.clip(u_prev + du[:self.n_inputs], self.u_min, self.u_max)

        return u
        
    

    '''
    def _matrix_augmentation(self, a, b, q, r, control_horizon, prediction_horizon):
      
        result_phi = numpy.zeros((self.n_states*prediction_horizon, self.n_states))        
        for n in range(prediction_horizon):
            ofs = n*self.n_states
            result_phi[ofs:ofs + self.n_states, :] = numpy.linalg.matrix_power(a, n+1)

        result_omega = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs))        
        for n in range(prediction_horizon):

            tmp = b.copy()
            for i in range(n):
                tmp+= numpy.linalg.matrix_power(a, i+1)@b

            ofs = n*self.n_states    
            result_omega[ofs:ofs + self.n_states, :] = tmp



        result_theta = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs*prediction_horizon))
        for n in range(prediction_horizon):
            for m in range(control_horizon):
                tmp = b.copy()
                for i in range(n):
                    tmp+= numpy.linalg.matrix_power(a, i+1)@b

                ofs_n = n*self.n_states   
                ofs_m = m*self.n_inputs 
                result_theta[ofs_n:ofs_n + self.n_states, ofs_m:ofs_m + self.n_inputs] = tmp


        result_q_aug = numpy.zeros((self.n_states*prediction_horizon, self.n_states*prediction_horizon))
        for n in range(prediction_horizon):
            ofs = n*self.n_states
            result_q_aug[ofs:ofs+self.n_states, ofs:ofs+self.n_states] = q

        
        result_r_aug = numpy.zeros((self.n_inputs*prediction_horizon, self.n_inputs*prediction_horizon))
        for n in range(prediction_horizon):
            ofs = n*self.n_inputs
            result_r_aug[ofs:ofs+self.n_inputs, ofs:ofs+self.n_inputs] = r
   
        return result_phi, result_omega, result_theta, result_q_aug, result_r_aug
    '''


    def _matrix_augmentation(self, a, b, q, r, control_horizon, prediction_horizon):
        # Precompute powers of A to avoid recalculating
        a_powers = [numpy.linalg.matrix_power(a, i+1) for i in range(prediction_horizon)]

        # Construct result_phi matrix
        result_phi = numpy.zeros((self.n_states * prediction_horizon, self.n_states))        
        for n in range(prediction_horizon):
            ofs = n * self.n_states
            result_phi[ofs:ofs + self.n_states, :] = a_powers[n]

        # Construct result_omega matrix
        result_omega = numpy.zeros((self.n_states * prediction_horizon, self.n_inputs))        
        for n in range(prediction_horizon):
            tmp = numpy.zeros_like(b)
            for i in range(n + 1):  # Cumulative effect up to n+1
                tmp += a_powers[i] @ b
            ofs = n * self.n_states    
            result_omega[ofs:ofs + self.n_states, :] = tmp

        # Construct result_theta matrix with control horizon limit
        result_theta = numpy.zeros((self.n_states * prediction_horizon, self.n_inputs * control_horizon))
        for n in range(prediction_horizon):
            for m in range(min(control_horizon, n + 1)):
                tmp = numpy.zeros_like(b)
                for i in range(n - m):  # Only sum up to the appropriate horizon
                    tmp += a_powers[i] @ b
                ofs_n = n * self.n_states   
                ofs_m = m * self.n_inputs 
                result_theta[ofs_n:ofs_n + self.n_states, ofs_m:ofs_m + self.n_inputs] = tmp

        # Construct augmented Q matrix
        result_q_aug = numpy.zeros((self.n_states * prediction_horizon, self.n_states * prediction_horizon))
        for n in range(prediction_horizon):
            ofs = n * self.n_states
            result_q_aug[ofs:ofs + self.n_states, ofs:ofs + self.n_states] = q

        # Construct augmented R matrix
        result_r_aug = numpy.zeros((self.n_inputs * control_horizon, self.n_inputs * control_horizon))
        for n in range(control_horizon):
            ofs = n * self.n_inputs
            result_r_aug[ofs:ofs + self.n_inputs, ofs:ofs + self.n_inputs] = r

        return result_phi, result_omega, result_theta, result_q_aug, result_r_aug
        