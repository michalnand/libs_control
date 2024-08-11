import numpy

'''
solve MPC controller for discrete discrete system
x(n+1) = Ax(n) + Bu(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)
''' 
class MPC: 

    def __init__(self, a, b, q, r, control_horizon, prediction_horizon, antiwindup = 10**10):
        self.control_horizon    = control_horizon
        self.prediction_horizon = prediction_horizon
        self.antiwindup         = antiwindup

        self.n_states = a.shape[0]
        self.n_inputs = b.shape[1]  

        self.phi, self.omega, theta, q_aug, r_aug = self._matrix_augmentation(a, b, q, r, control_horizon, prediction_horizon)

        
        h = (theta.T@q_aug@theta) + r_aug
        sigma = numpy.linalg.inv(h)@theta.T@q_aug

        #cut only part for u(n) (we dont care to compute future u(n+1...))
        self.sigma = sigma[0:self.n_inputs, :]

        self.xr_aug = numpy.zeros((self.n_states*self.prediction_horizon, 1))

      
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

        '''
        result_theta = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs*prediction_horizon))
        for m in range(prediction_horizon):
            tmp_n = min(m, control_horizon)
            for n in range(tmp_n): 
                a_pow = m - n

                ofs_m = m*self.n_states
                ofs_n = n*self.n_inputs
                
                tmp = b.copy()
                for i in range(a_pow):  
                    tmp+= numpy.linalg.matrix_power(a, i+1)@b

                result_theta[ofs_m:ofs_m + self.n_states, ofs_n:ofs_n + self.n_inputs] = tmp
        '''

        '''
        result_theta = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs*prediction_horizon))
        for n in range(prediction_horizon):

            tmp = b.copy()
            for i in range(n):
                tmp+= numpy.linalg.matrix_power(a, i+1)@b

            ofs = n*self.n_states    
            result_theta[ofs:ofs + self.n_states, 0:self.n_inputs] = tmp
        '''


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
    
    
    def forward(self, xr, x, u_prev):   
        #tile xr into xr_aug
        for n in range(self.prediction_horizon):
            ofs = n*self.n_states
            self.xr_aug[ofs:ofs + self.n_states, :] = xr[n, :]


        return self.forward_trajectory(xr, x, u_prev)
    

    def forward_trajectory(self, xr, x, u_prev):   
        self.xr_aug = numpy.reshape(xr, (xr.shape[0]*xr.shape[1], 1))

        s  = self.xr_aug - self.phi@x - self.omega@u_prev
        du = self.sigma@s
        u  = numpy.clip(u_prev + du, -self.antiwindup, self.antiwindup)

        return u
    
    
    

if __name__ == "__main__":
    n_states = 4
    n_inputs = 2

    a = numpy.random.randn(n_states, n_states)
    b = numpy.random.randn(n_states, n_inputs)

    q = numpy.eye(n_states, n_states)
    r = numpy.eye(n_inputs, n_inputs)

    mpc = MPC(a, b, q, r, 1, 64)

    


    u_prev = numpy.random.randn(n_inputs, 1)

    for i in range(10):
        xr = numpy.random.randn(n_states, 1)
        x  = numpy.random.randn(n_states, 1)
    
        u_new = mpc.forward(xr, x, u_prev)
        u_prev = u_new.copy()

        print(u_new.shape)
    