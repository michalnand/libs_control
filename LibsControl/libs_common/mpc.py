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

    def __init__(self, a, b, q, r, prediction_horizon, antiwindup = 10**10):
        self.prediction_horizon = prediction_horizon
        self.antiwindup         = antiwindup

        self.n_states = a.shape[0]
        self.n_inputs = b.shape[1]  

        self.phi, self.omega, theta, q_aug, r_aug = self._matrix_augmentation(a, b, q, r, prediction_horizon)

        
        h = (theta.T@q_aug@theta) + r_aug
        sigma = numpy.linalg.inv(h)@theta.T@q_aug

        #cut only part for u(n) (we dont care to compute future u(n+1...))
        self.sigma = sigma[0:self.n_inputs, :]


    
    def _matrix_augmentation(self, a, b, q, r, prediction_horizon):
      
        result_phi = numpy.zeros((self.n_states*prediction_horizon, self.n_states))        
        for n in range(prediction_horizon):
            ofs = n*self.n_states
            result_phi[ofs:ofs + self.n_states, :] = numpy.linalg.matrix_power(a, n+1)

        print("result_phi = ", result_phi.shape)

        result_omega = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs))        
        for n in range(prediction_horizon):
            ofs = n*self.n_states    

            tmp = b.copy()
            for i in range(n):
                tmp+= numpy.linalg.matrix_power(a, i+1)@b

            result_omega[ofs:ofs + self.n_states, :] = tmp



        print("result_omega = ", result_omega.shape)


        result_theta = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs*prediction_horizon))
        for m in range(prediction_horizon):
            for n in range(m + 1):
                a_pow = m - n

                ofs_m = m*self.n_states
                ofs_n = n*self.n_inputs
                
                tmp = b.copy()
                for i in range(a_pow):  
                    tmp+= numpy.linalg.matrix_power(a, i+1)@b

                result_theta[ofs_m:ofs_m + self.n_states, ofs_n:ofs_n + self.n_inputs] = tmp


        print("result_theta = ", result_theta.shape)

           
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
        
        xr_aug = numpy.zeros((self.n_states*self.prediction_horizon, 1))
        for n in range(self.prediction_horizon):
            ofs = n*self.n_states
            xr_aug[ofs:ofs + self.n_states, :] = xr.copy()

        s  = xr_aug - self.phi@x - self.omega@u_prev
        du = self.sigma@s
        u  = u_prev + du

        return u
    
    def test(self):
        
        #matrices test
        u_initial = numpy.random.randn(self.n_inputs, 1)
        du        = numpy.random.randn(self.prediction_horizon, self.n_inputs, 1) 
        x_initial = numpy.random.randn(self.n_states, 1)
        
        
        x_result = []

        u  = u_initial.copy()
        x  = x_initial.copy()
        for n in range(self.prediction_horizon):
            u = u + du[n]
            x = a@x + b@u
            x_result.append(x)

        x_result = numpy.array(x_result)


        du_flat = du.reshape(self.prediction_horizon*self.n_inputs, 1)
        x_pred = self.phi@x_initial + self.omega@u_initial + self.theta@du_flat
        x_pred = x_pred.reshape(self.prediction_horizon, n_states, 1)


        error = ((x_result - x_pred)**2).mean(axis=(1, 2))

        print(error)

        print("mean error = ", error.mean())
        
    

if __name__ == "__main__":
    n_states = 7
    n_inputs = 3

    a = numpy.random.randn(n_states, n_states)
    b = numpy.random.randn(n_states, n_inputs)

    q = numpy.eye(n_states, n_states)
    r = numpy.eye(n_inputs, n_inputs)

    mpc = MPC(a, b, q, r, 10)

    


    u_prev = numpy.random.randn(n_inputs, 1)

    for i in range(10):
        xr = numpy.random.randn(n_states, 1)
        x  = numpy.random.randn(n_states, 1)
    
        u_new = mpc.forward(xr, x, u_prev)
        u_prev = u_new.copy()

        print(u_new.shape)
    