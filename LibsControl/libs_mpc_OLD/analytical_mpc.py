import numpy
import scipy

"""
    gradient based model predictive control MPC

    system to control :
        x(n+1) = Ax(n) + Bu(n)
        u(n+1) = u(n) + du(n)

    constrains : 
        u  = clip(u,  u_min, u_max)     : actuator saturation
"""

class AnalyticalMPC:

    def __init__(self, mat_a, mat_b, mat_q, mat_r, prediction_horizon = 16, control_horizon = 4, u_max = 10**10):
        self.n_states = mat_a.shape[0]
        self.n_inputs = mat_b.shape[1]  


        self.phi, self.omega, theta, q_aug, r_aug = self._matrix_augmentation(mat_a, mat_b, mat_q, mat_r, control_horizon, prediction_horizon)

        h = (theta.T@q_aug@theta) + r_aug
        sigma = numpy.linalg.inv(h)@theta.T@q_aug

        #cut only part for u(n) (we dont care to compute future u(n+1...))
        self.sigma = sigma[0:self.n_inputs, :]

        self.xr_aug = numpy.zeros((self.n_states*prediction_horizon, 1))

        self.u_max = u_max

    def forward(self, xr, x):   
        self.xr_aug = numpy.reshape(xr, (xr.shape[0]*xr.shape[1], 1))


        s = self.xr_aug - self.phi@x
        u = self.sigma@s
        
        u  = numpy.clip(u, -self.u_max, self.u_max)
    
        return u
    


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






class AnalyticalMPCDirect:

    def __init__(self, mat_a, mat_b, mat_q, mat_r, prediction_horizon = 16, control_horizon = 4):
        self.n_states = mat_a.shape[0]
        self.n_inputs = mat_b.shape[1]  


        self.phi, self.omega, theta, q_aug, r_aug = self._matrix_augmentation(mat_a, mat_b, mat_q, mat_r, control_horizon, prediction_horizon)

        h = (theta.T@q_aug@theta) + r_aug
        sigma = numpy.linalg.inv(h)@theta.T@q_aug

        #cut only part for u(n) (we dont care to compute future u(n+1...))
        self.sigma = sigma[0:self.n_inputs, :]

        self.xr_aug = numpy.zeros((self.n_states*prediction_horizon, 1))


        self.u_min = -1.0
        self.u_max =  1.0
        self.du_min = -1.0
        self.du_max = 1.0

    def forward(self, xr, x, u_prev, contrains_func = None):   
        self.xr_aug = numpy.reshape(xr, (xr.shape[0]*xr.shape[1], 1))

        s  = self.xr_aug - self.phi@x - self.omega@u_prev
        du = self.sigma@s
        du = numpy.clip(du, self.du_min, self.du_max)

        if contrains_func is not None:
            u  = contrains_func(u_prev + du)
        else:
            u  = numpy.clip(u_prev + du, self.u_min, self.u_max)

        return u
    


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
        