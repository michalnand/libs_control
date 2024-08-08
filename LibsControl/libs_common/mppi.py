import numpy
from .lqr_discrete import *

'''
solve MPC controller for discrete discrete system
x(n+1) = Ax(n) + Bu(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)
''' 
class MPPI: 

    def __init__(self, a, b, q, r, prediction_horizon, antiwindup = 10**10):
        self.prediction_horizon = prediction_horizon
        self.antiwindup         = antiwindup

        self.q = q
        self.r = r

        self.n_states = a.shape[0]
        self.n_inputs = b.shape[1] 

        self.a = a
        self.b = b 

        self.lqr = LQRDiscrete(a, b, q, r, antiwindup)


    def _softmax(self, x, dim):
        x_norm = x - numpy.max(x, axis=dim, keepdims=True)
        w = numpy.exp(x_norm)

        w = w/(w.sum(axis=dim, keepdims=True) + 10**-6)
        
        return w
    

    def update_step(self, xr, x, u_initial, n_rollouts):
        
        # set initial state
        xr_pred = numpy.zeros((self.prediction_horizon, n_rollouts, self.n_states))
        xr_pred[0, :] = x[:, 0]
        
        # set initial u
        u_pred = numpy.expand_dims(u_initial, 1) + numpy.random.randn(self.prediction_horizon, n_rollouts, self.n_inputs)

        '''
        costs  = numpy.zeros((self.prediction_horizon, n_rollouts))
        
        # predict trajectory
        for n in range(self.prediction_horizon-1):
            j = (((xr[n] - xr_pred[n])**2)@self.q).sum(axis=1)
            j+= ((u_pred[n]**2)@self.r).sum(axis=1)
            
            costs[n] = j

            xr_pred[n+1] = xr_pred[n]@self.a.T + u_pred[n]@self.b.T
        
        w = self._softmax(-costs, 1)
        w = numpy.expand_dims(w, 2)
        '''
        
        
        costs  = numpy.zeros(n_rollouts)        
        # predict trajectory
        for n in range(self.prediction_horizon-1):
            j = (((xr[n] - xr_pred[n])**2)@self.q).sum(axis=1)
            j+= ((u_pred[n]**2)@self.r).sum(axis=1)
            
            costs+= j

            xr_pred[n+1] = xr_pred[n]@self.a.T + u_pred[n]@self.b.T
        

        w = self._softmax(-costs, 0)
        w = numpy.expand_dims(w, 0)
        w = numpy.expand_dims(w, 2)
        
        u_result = (u_pred*w).sum(axis=1)
        
        return u_result

    '''
    def forward_trajectory(self, xr, x, u_prev):   

        u_new       = numpy.zeros((self.prediction_horizon, self.n_inputs))
        #u_new[:]    = u_prev[:, 0].copy()

        iterations     = 10
        rollouts_count = 64
        for i in range(iterations):
            u_new = self.update_step(xr, x, u_new, rollouts_count)


        u  = numpy.expand_dims(u_new[0, :], 1)
        u  = numpy.clip(u, -self.antiwindup, self.antiwindup)
        
        return u
    '''

    def forward_trajectory(self, xr, x, integral_action):  
        xr_tmp = xr[0]
        xr_tmp = numpy.expand_dims(xr_tmp, 1)

        # we use LQR for initial u value guess
        u, integral_action_new = self.lqr.forward(xr_tmp, x, integral_action)

        u_new       = numpy.zeros((self.prediction_horizon, self.n_inputs))
        #u_new[:]    = u[:, 0].copy()

        iterations = 20
        for i in range(iterations):
            u_new = self.update_step(xr, x, u_new, 256)

        u  = numpy.expand_dims(u_new[0, :], 1)
        u  = numpy.clip(u, -self.antiwindup, self.antiwindup)
      

        return u, integral_action_new

    
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
    n_states = 4
    n_inputs = 2

    a = numpy.random.randn(n_states, n_states)
    b = numpy.random.randn(n_states, n_inputs)

    q = numpy.eye(n_states, n_states)
    r = numpy.eye(n_inputs, n_inputs)

    mpc = MPPI(a, b, q, r, 64)

    


    u_prev = numpy.random.randn(n_inputs, 1)

    for i in range(10):
        xr = numpy.random.randn(n_states, 1)
        x  = numpy.random.randn(n_states, 1)
    
        u_new = mpc.forward(xr, x, u_prev)
        u_prev = u_new.copy()

        print(u_new.shape)
    