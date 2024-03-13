import numpy

'''
solve LQG controller for contunuous discrete system
x(n+1) = Ax(n) + Bu(n)
y(n) = Cx(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
C matrix, shape (n_outputs, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)
''' 
class MPC:

    def __init__(self, a, b, q, r, prediction_horizon, antiwindup = 10**10):
        self.prediction_horizon = prediction_horizon
        self.n_states = a.shape[0]
        self.n_inputs = b.shape[1]

        self.result_o, result_m, result_q_aug, result_r_aug = self._create_lifted_matrices(a, b, q, r, prediction_horizon)


        tmp  = result_m.T@result_q_aug@result_m + result_r_aug
        tmp  = numpy.linalg.inv(tmp)

        self.control_mat = tmp@result_m.T@result_q_aug



    def _create_lifted_matrices(self, a, b, q, r, prediction_horizon):
      

        result_o = numpy.zeros((self.n_states*prediction_horizon, self.n_states))        
        for n in range(prediction_horizon):
            ofs = n*self.n_states
            
            #result_o[ofs:ofs + self.n_states, :] = a**(n+1)
            result_o[ofs:ofs + self.n_states, :] = numpy.linalg.matrix_power(a, n+1)

       
        result_m = numpy.zeros((self.n_states*prediction_horizon, self.n_inputs*prediction_horizon))

        for m in range(prediction_horizon):
            for n in range(prediction_horizon):
                if m >= n:
                    a_pow = max(m - n, 0)
                    
                    ofs_m = m*self.n_states
                    ofs_n = n*self.n_inputs
                    
                    #result_m[ofs_m:ofs_m + self.n_states, ofs_n:ofs_n + self.n_inputs] = (a**a_pow)@b
                    result_m[ofs_m:ofs_m + self.n_states, ofs_n:ofs_n + self.n_inputs] = numpy.linalg.matrix_power(a, a_pow)@b
                    


        result_q_aug = numpy.zeros((self.n_states*prediction_horizon, self.n_states*prediction_horizon))
        for n in range(prediction_horizon):
            ofs = n*self.n_states
            result_q_aug[ofs:ofs+self.n_states, ofs:ofs+self.n_states] = q

        result_r_aug = numpy.zeros((self.n_inputs*prediction_horizon, self.n_inputs*prediction_horizon))
        for n in range(prediction_horizon):
            ofs = n*self.n_inputs
            result_r_aug[ofs:ofs+self.n_inputs, ofs:ofs+self.n_inputs] = r

        
        return result_o, result_m, result_q_aug, result_r_aug
    
    def forward(self, xr, x):   
        
        xr_aug = numpy.zeros((self.n_states*self.prediction_horizon, 1))
        for n in range(self.prediction_horizon):
            ofs = n*self.n_states
            xr_aug[ofs:ofs + self.n_states, :] = xr

        e = xr_aug - self.result_o@x

        '''
        u = self.control_mat@e
        u = u[0:self.n_inputs, :]
        '''
        #compute only one step forward
        u = self.control_mat[0:self.n_inputs, :]@e

        return u
        
    
    

if __name__ == "__main__":
    n_states = 4
    n_inputs = 2

    a = numpy.random.rand(n_states, n_states)
    b = numpy.random.rand(n_states, n_inputs)

    q = numpy.eye(n_states, n_states)
    r = numpy.eye(n_inputs, n_inputs)

    mpc = MPC(a, b, q, r, 32)

    xr = numpy.random.randn(n_states, 1)
    x  = numpy.random.randn(n_states, 1)

    mpc.forward(xr, x)