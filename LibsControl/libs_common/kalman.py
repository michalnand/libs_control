import numpy
import matplotlib.pyplot as plt
import scipy.linalg

'''
constant velocity model Kalman filter
'''

  
class KalmanFilterVel:
    
    def __init__(self, n_count, r, q = 10**-4):
        
        mat_a = [
            [1.0, 1.0],
            [0.0, 1.0]
        ]
        
        self.mat_a = numpy.array(mat_a)

        n = self.mat_a.shape[0]

        c = numpy.eye(n)


        q_mat = q*numpy.eye(n) 
       
        r_mat = numpy.zeros((n, n))
        for i in range(n):
            r_mat[i][i] = r*(2**i)

        self.mat_k = self._find_kalman_gain(self.mat_a, c, r_mat, q_mat)


        self.x_hat = numpy.zeros((n_count, n)) 

        
        print("kalman gain matrix")
        print(self.mat_k)
        print("\n\n")

        self.x0 = numpy.zeros(n_count)
        self.x1 = numpy.zeros(n_count)
        self.x2 = numpy.zeros(n_count)

    
    def step(self, x_measurement, return_full_state = False):

        self.x2 = self.x1.copy()
        self.x1 = self.x0.copy()
        self.x0 = x_measurement.copy()

        x = self.x0
        v = self.x0 - self.x1

        x_all = numpy.vstack([x, v]).T

        error = x_all - self.x_hat

        self.x_hat  = self.x_hat@self.mat_a.T + error@self.mat_k.T

        if return_full_state:
            return self.x_hat
        else:
            return self.x_hat[:, 0]
        
    def predict(self, num_steps, return_full_state = False):
        x_result  = numpy.zeros((num_steps, ) + self.x_hat.shape)

        x_hat = self.x_hat.copy()

        for n in range(num_steps):
            x_hat       = x_hat@self.mat_a.T
            x_result[n] = x_hat
                    
        if return_full_state:
            return x_result
        else:
            return x_result[:, :, 0]
    
    def _find_kalman_gain(self, a, c, r, q):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        k = p@c.T@scipy.linalg.inv(c@p@c.T + r)
        return k


    
class KalmanFilterACC:
    
    def __init__(self, n_count, r, q = 10**-4):
        
        mat_a = [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ]
        
        self.mat_a = numpy.array(mat_a)

        n = self.mat_a.shape[0]

        c = numpy.eye(n)


        q_mat = q*numpy.eye(n) 
       
        r_mat = numpy.zeros((n, n))
        for i in range(n):
            r_mat[i][i] = r*(2**i)

        self.mat_k = self._find_kalman_gain(self.mat_a, c, r_mat, q_mat)


        self.x_hat = numpy.zeros((n_count, n)) 

        
        print("kalman gain matrix")
        print(self.mat_k)
        print("\n\n")

        self.x0 = numpy.zeros(n_count)
        self.x1 = numpy.zeros(n_count)
        self.x2 = numpy.zeros(n_count)

    
    def step(self, x_measurement, return_full_state = False):

        self.x2 = self.x1.copy()
        self.x1 = self.x0.copy()
        self.x0 = x_measurement.copy()

        x = self.x0
        v = self.x0 - self.x1
        a = self.x0 - 2*self.x1 + self.x2

        x_all = numpy.vstack([x, v, a]).T

        error = x_all - self.x_hat

        self.x_hat  = self.x_hat@self.mat_a.T + error@self.mat_k.T

        if return_full_state:
            return self.x_hat
        else:
            return self.x_hat[:, 0]
        
    def predict(self, num_steps, return_full_state = False):
        x_result  = numpy.zeros((num_steps, ) + self.x_hat.shape)

        x_hat = self.x_hat.copy()

        for n in range(num_steps):
            x_hat       = x_hat@self.mat_a.T
            x_result[n] = x_hat
                    
        if return_full_state:
            return x_result
        else:
            return x_result[:, :, 0]
    
    def _find_kalman_gain(self, a, c, r, q):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        k = p@c.T@scipy.linalg.inv(c@p@c.T + r)
        return k





if __name__ == "__main__":


    #num of steps
    n_steps = 1000

    #noise variance
    rx = 0.05

    n_count = 10

    #1D kalman
    #filter = KalmanFilter((1, ), rx)
    filter = KalmanFilterACC(n_count, rx, q = 10**-8)

    #reference - simple low pass filter
    x_lp = 0.0

    #process state variables
    x_true = 0.0
    dx_true= 0.01
    
    #log result
    x_true_result   = numpy.zeros(n_steps)
    x_measurment_result = numpy.zeros(n_steps)
    x_kalman_result = numpy.zeros(n_steps)
    x_lp_result     = numpy.zeros(n_steps)

    for i in range(n_steps):
       

        if (i+1)%200 == 0:
            dx_true*= -1

        x_true+= dx_true

        z_measurement = x_true + (rx**0.5)*numpy.random.randn()

        x_kalman = filter.step(numpy.ones(n_count)*z_measurement)

        k = 0.95
        x_lp = k*x_lp + (1.0 - k)*z_measurement

        x_true_result[i]        = x_true
        x_measurment_result[i]  = z_measurement
        x_kalman_result[i]      = x_kalman[0].item()
        x_lp_result[i]          = x_lp

    

    snr_raw = 10.0*numpy.log10( (x_true_result**2)/( (x_measurment_result - x_true_result)**2  ) )
    snr_raw = snr_raw.mean()

    print("snr_raw = ", round(snr_raw, 3), "dB")


    snr_kalman = 10.0*numpy.log10( (x_true_result**2)/( (x_kalman_result - x_true_result)**2  ) )
    snr_kalman = snr_kalman.mean()

    print("snr_kalman = ", round(snr_kalman, 3), "dB")


    snr_lp = 10.0*numpy.log10( (x_true_result**2)/( (x_lp_result - x_true_result)**2  ) )
    snr_lp = snr_lp.mean()

    print("snr_lp = ", round(snr_lp, 3), "dB")

   

    plt.clf()
    plt.tight_layout()
    plt.xlabel("step")
    plt.ylabel("value")
    plt.plot(x_true_result, color='red', label="true value", linewidth=3.0)
    plt.plot(x_measurment_result, color='salmon', label="noised measurement", alpha=0.5)
    plt.plot(x_lp_result, color='lime', label="low pass filter estimated", linewidth=1.5)
    plt.plot(x_kalman_result, color='blue', label="kalman filter estimated", linewidth=1.5)

    plt.legend()
    plt.show()
    