import torch
import numpy
import matplotlib.pyplot as plt

'''
class KalmanFilter:

    def __init__(self, dims = 1):
        
        #initial guess
        self.x_hat = torch.zeros(dims)
        self.p     = 0.5*torch.ones(dims)

    def step(self, z, dz, r):
        #kalman gain
        k      = self.p/(self.p + r)

        #estimating the current state
        self.x_hat  = self.x_hat + k*(z - self.x_hat)
 
        self.x_hat  = self.x_hat + dz

        #update the current estimate uncertainty
        self.p      = (1.0 - k)*self.p
        
        return self.x_hat
'''


class KalmanFilter:

    def __init__(self, dims = 1):
        
        #initial guess
        self.x_hat = torch.zeros(dims)
        self.p     = 0.5*torch.ones(dims)

    '''
        z   - position measurement
        dz  - velocity measurement
        pz  - position measurement uncertaininty
        pdz - velocity measurement uncertaininty
    '''
    def step(self, z, dz, pz, pdz, dt = 1):
        #1, prediction
		#predict the state and uncertaininty
        self.x_hat  = self.x_hat + dz*dt
        self.p      = self.p + pdz*(dt**2)

        #2, kalman gain
        k      = self.p/(self.p + pz)

        #3, update
        self.x_hat  = self.x_hat + k*(z - self.x_hat)
        self.p      = (1.0 - k)*self.p
        
        return self.x_hat



'''
if __name__ == "__main__":

    n       = 400
    dims    = 1

    pz      = 0.1
    pdz     = 0.001

    dt      = 1.0


    z_true      = 0.0
    dz_true     = 0.03


    filter = KalmanFilter(dims)

    t_ = numpy.zeros(n)
    z_ = numpy.zeros(n)
    dz_ = numpy.zeros(n)
    x_kalman_ = numpy.zeros(n)

    for i in range(n):
        if (i+1)%100 == 0:
            dz_true*= -1

        z_true+= dz_true*dt


        z   = z_true    + (pz**0.5)*torch.randn(dims)
        dz  = dz_true   + (pdz**0.5)*torch.randn(dims)

        x = filter.step(z, dz, pz, pdz, dt)

        t_[i]           = z_true
        z_[i]           = z
        dz_[i]          = dz
        x_kalman_[i]    = x

    
    snr_raw = 10.0*numpy.log10( (t_**2)/( (z_ - t_)**2  ) )
    snr_raw = snr_raw.mean()

    snr_fil = 10.0*numpy.log10( (t_**2)/( (x_kalman_ - t_)**2  ) )
    snr_fil = snr_fil.mean()

    print(">>> snr_raw = ", snr_raw)
    print(">>> snr_fil = ", snr_fil)

    plt.clf()
    plt.xlabel("time")
    plt.ylabel("position")
    plt.plot(t_, color='red', label="position true value")
    plt.plot(z_, color='green', label="position measurement")
    plt.plot(dz_, color='lime', label="velocity measurement")
    plt.plot(x_kalman_, color='blue', label="kalman estimated")

    plt.legend()
    plt.show()
'''


if __name__ == "__main__":

    n       = 1000
    dims    = 1

    pz      = 0.1
    pdz     = 0.00001

    dt      = 1.0


    z_true      = 0.0
    dz_true     = 0.03

    x_lp        = 0.0


    filter = KalmanFilter(dims)

    t_ = numpy.zeros(n)
    z_ = numpy.zeros(n)
    x_kalman_ = numpy.zeros(n)
    x_lp_     = numpy.zeros(n)

    for i in range(n):
       
        z_true = 1.0

        z   = z_true    + (pz**0.5)*torch.randn(dims)

        x = filter.step(z, 0, pz, pdz, dt)

        k = 0.95
        x_lp = k*x_lp + (1.0 - k)*z

        t_[i]           = z_true
        z_[i]           = z
        x_kalman_[i]    = x
        x_lp_[i]        = x_lp

    
    snr_raw = 10.0*numpy.log10( (t_**2)/( (z_ - t_)**2  ) )
    snr_raw = snr_raw.mean()

    snr_fil_kalman = 10.0*numpy.log10( (t_**2)/( (x_kalman_ - t_)**2  ) )
    snr_fil_kalman = snr_fil_kalman.mean()

    snr_fil_lp = 10.0*numpy.log10( (t_**2)/( (x_lp_ - t_)**2  ) )
    snr_fil_lp = snr_fil_lp.mean()

    print(">>> snr_raw = ", snr_raw)
    print(">>> snr_fil_kalman = ", snr_fil_kalman)
    print(">>> snr_fil_lp = ", snr_fil_lp)

    plt.clf()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.plot(t_, color='red', label="true value", linewidth=3.0)
    plt.plot(z_, color='darkred', label="noised measurement", alpha=0.5)
    plt.plot(x_lp_, color='green', label="low pass estimated", linewidth=2.0)
    plt.plot(x_kalman_, color='blue', label="kalman estimated", linewidth=2.0)

    plt.legend()
    plt.show()