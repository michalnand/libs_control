import numpy
import scipy.linalg


'''
solve LQR for discrete dynamical system : 
x(n+1) = A@x(n) + B@u(n)
y  = x

controller : 
e_int = e_int + (xr - x)d*t
u = -K@x + Kie_int
 
where : 
N : system order
M : system inputs count
K : ssytem outputs

A : system dynamic matrix, NxN
B : system input   matrix, NxM
C : diagonal matrix, KxN, projecting state x to output y

x   : state column vector, Nx1
x_r : required state, same shape as x
u   : control input, column vector, Mx1

Q : diagonal matrix, NxN, weighting for required state elements
R : diagonal amtrix, MxM, weighting for controll value

returns : 

K  : computed controller feedback matrix, NxM
Ki : computed gain matrix for integral acttion, NxM
'''
class LQRIDiscreteSolver:

    def __init__(self, a, b, q, r):

        self.a = a
        self.b = b

        self.q = q
        self.r = r


    def copy(self):
        result = LQRIDiscreteSolver(self.a, self.b, self.q, self.r)

        result.k  = self.k.copy()
        result.ki = self.ki.copy()

        return result
        
    def solve(self):
        self.k, self.ki = self._find_ki(self.a, self.b, self.q, self.r)

        
        return self.k, self.ki
    
    def closed_loop_response(self, xr, steps = 500, noise = 0.0, disturbance = False):

        n = self.a.shape[0]  #system order
        m = self.b.shape[1]  #inputs count
        k = self.a.shape[0]  #outputs count


        u_result = numpy.zeros((steps, m))
        x_result = numpy.zeros((steps, n))

        x  = numpy.zeros((n, 1))
        error_sum = numpy.zeros((1, m))

        for n in range(steps):
            x_obs       = x + noise*numpy.random.randn(x.shape[0], x.shape[1])

            u, error_sum = self.forward(xr, x_obs, error_sum)
    
            #apply disturbance
            if disturbance == True and n >= steps//2:
                u+= 1 

            #system dynamics step
            x     = x + (self.a@x + self.b@u)
            

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]
            

        return u_result, x_result
             

    def forward(self, xr, x, error_sum):
        #compute error
        error     = xr - x

        #integral action
        error_sum_new = error_sum + error

        #apply controll law
        u = -self.k@x + self.ki@error_sum_new

        return u, error_sum_new

    
    '''
    solve the continuous time lqr controller.
    dx = A x + B u
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''
    def _find_ki(self, a, b, q, r):

        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count
        k = a.shape[0]  #outputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+k, n+k))
        b_aug = numpy.zeros((n+k, m))
        q_aug = numpy.zeros((n+k, n+k))

        a_aug[0:n, 0:n] = a
        #a_aug[n:, 0:n]  = c

        b_aug[0:n,0:m]  = b

        for i in range(n):
            a_aug[i + n, i]     = 1.0
            a_aug[i + n, i + n] = 1.0

        #project Q matric to output, and fill augmented q matrix
        #tmp = (c@q).sum(axis=1)
        for i in range(k):
            q_aug[n+i][n+i] = q[i, i]

        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain 
        ki_tmp =  scipy.linalg.inv(r)@(b_aug.T@p)

        ki_tmp[numpy.abs(ki_tmp) < 10**-10] = 0

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki
    