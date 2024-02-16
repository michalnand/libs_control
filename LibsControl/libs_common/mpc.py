import numpy
import scipy
import torch

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
        self.q = torch.from_numpy(q).float()
        self.r = torch.from_numpy(r).float()

        self.a = torch.from_numpy(a).float()
        self.b = torch.from_numpy(b).float()

        self.prediction_horizon = prediction_horizon
        self.antiwindup = antiwindup

        u_initial = torch.zeros((prediction_horizon, self.b.shape[1], 1), dtype=torch.float32)
        self.u = torch.nn.Parameter(u_initial, requires_grad = True) 

        self.optimizer = torch.optim.Adam([self.u], lr=0.1)


    '''
    inputs:
        xr : required state, shape (n_states, 1)
        y  : system output, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)
        x_hat : storage for estimated full state, shape (n_states, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
        x_hat_new : storage for estimated full state, shape (n_states, 1)
    '''
    def forward(self, xr, x):   
        n_max = 50

        for n in range(n_max):
            
            loss = 0.0

            xr_t =  torch.from_numpy(xr).detach().float()
            x_t  = torch.from_numpy(x).detach().float()

            for t in range(self.prediction_horizon):
                x_t = self.a@x_t + self.b@self.u[t] 

            loss+= (self.u[0].T@self.r@self.u[0]).mean()
            loss+= ((xr_t - x_t)**2).mean()

            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        print(loss, self.u[0])

        return self.u[0].detach().numpy()


    '''
    compute kalman gain matrix F for observer : 
    x_hat(n+1) = Ax_hat(n) + Bu(n) + F(y(n) - Cx_hat(n))
    ''' 
    def solve_kalman_gain(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        f = p@c.T@scipy.linalg.inv(c@p@c.T + r)

        return f