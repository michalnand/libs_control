import numpy
import matplotlib.pyplot as plt


'''
constant velocity model Kalman filter
'''
class KalmanFilter:
    # shape - filter process batched element wise, with given shape
    # rx - position noise variance
    # q  - process noise variance
    def __init__(self, shape, rx, q = 10**-3):

        self.x0 = numpy.zeros(shape)
        self.x1 = numpy.zeros(shape)

        #position variance
        self.rx = rx

        #velocity variance
        self.rv = 2*rx

        self.q  = q

        self.x_hat = numpy.zeros(shape)
        self.v_hat = numpy.zeros(shape)

        #initial uncertainity
        self.px = 1.0*numpy.ones(shape)
        self.pv = 1.0*numpy.ones(shape)

    # x - noised position measurement
    # returns denoised position and velocity
    def step(self, x_measurement):
        self.x1 = self.x0.copy()
        self.x0 = x_measurement


        #state predict
        self.x_hat = self.x_hat + self.v_hat
        self.v_hat = self.v_hat
        self.px = self.px + self.pv
        self.pv = self.pv + self.q

        #kalman gain
        kx = self.px/(self.px + self.rx)
        kv = self.pv/(self.pv + self.rv)

        #update
        x = self.x0
        v = self.x0 - self.x1

        self.x_hat = self.x_hat + kx*(x - self.x_hat)
        self.v_hat = self.v_hat + kv*(v - self.v_hat)
        self.px = (1.0 - kx)*self.px
        self.pv = (1.0 - kv)*self.pv

        return self.x_hat, self.px
    
    #predixt n-steps into future, from given x_measurement as initial state
    def predict(self, num_steps):
        x_result  = numpy.zeros((num_steps, ) + self.x0.shape)
        px_result = numpy.zeros((num_steps, ) + self.x0.shape)

        x_hat = self.x_hat.copy()
        v_hat = self.v_hat.copy()

        px    = self.px.copy()
        pv    = self.pv.copy()

        for n in range(num_steps):
            x_hat = x_hat + v_hat
            v_hat = v_hat

            px    = px + pv
            pv    = pv + self.q

            x_result[n]     = x_hat
            px_result[n]    = px

        return x_result, px_result





'''
constant acceleration model Kalman filter
'''
class KalmanFilterACC:
    # shape - filter process batched element wise, with given shape
    # rx - position noise variance
    # q  - process noise variance
    def __init__(self, shape, rx, q = 10**-3):

        self.x0 = numpy.zeros(shape)
        self.x1 = numpy.zeros(shape)
        self.x2 = numpy.zeros(shape)

        #position variance
        self.rx = rx

        #velocity variance
        self.rv = 2*rx

        #acceleration variance
        self.ra = 4*rx


        self.q  = q

        self.x_hat = numpy.zeros(shape)
        self.v_hat = numpy.zeros(shape)
        self.a_hat = numpy.zeros(shape)

        #initial uncertainity
        self.px = 1.0*numpy.ones(shape)
        self.pv = 1.0*numpy.ones(shape)
        self.pa = 1.0*numpy.ones(shape)

    # x - noised position measurement
    # returns denoised position and velocity
    def step(self, x_measurement):
        self.x2 = self.x1.copy()
        self.x1 = self.x0.copy()
        self.x0 = x_measurement.copy()


        #state predict
        self.x_hat = self.x_hat + self.v_hat
        self.v_hat = self.v_hat + self.a_hat
        self.a_hat = self.a_hat

        self.px = self.px + self.pv
        self.pv = self.pv + self.pa
        self.pa = self.pa + self.q

        #kalman gain
        kx = self.px/(self.px + self.rx)
        kv = self.pv/(self.pv + self.rv)
        ka = self.pa/(self.pa + self.ra)

        #update
        x = self.x0
        v = self.x0 - self.x1
        a = self.x0 - 2.0*self.x1 + self.x2

        self.x_hat = self.x_hat + kx*(x - self.x_hat)
        self.v_hat = self.v_hat + kv*(v - self.v_hat)
        self.a_hat = self.a_hat + ka*(a - self.a_hat)

        self.px = (1.0 - kx)*self.px
        self.pv = (1.0 - kv)*self.pv
        self.pa = (1.0 - ka)*self.pa

        return self.x_hat, self.px
    
    #predixt n-steps into future, from given x_measurement as initial state
    def predict(self, num_steps):
        x_result  = numpy.zeros((num_steps, ) + self.x0.shape)
        px_result = numpy.zeros((num_steps, ) + self.x0.shape)

        x_hat = self.x_hat.copy()
        v_hat = self.v_hat.copy()
        a_hat = self.a_hat.copy()

        px    = self.px.copy()
        pv    = self.pv.copy()
        pa    = self.pa.copy()

        for n in range(num_steps):
            x_hat = x_hat + v_hat
            v_hat = v_hat + a_hat
            a_hat = a_hat

            px    = px + pv
            pv    = pv + pa
            pa    = pa + self.q

            x_result[n]     = x_hat
            px_result[n]    = px

        return x_result, px_result

if __name__ == "__main__":

    #num of steps
    n_steps = 1000

    #noise variance
    rx = 0.05

    #1D kalman
    filter = KalmanFilter((1, ), rx)

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

        x_kalman, _ = filter.step(numpy.array(z_measurement))

        k = 0.9
        x_lp = k*x_lp + (1.0 - k)*z_measurement

        x_true_result[i]        = x_true
        x_measurment_result[i]  = z_measurement
        x_kalman_result[i]      = x_kalman.item()
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
    plt.plot(x_true_result, color='red', label="true value", linewidth=2.0)
    plt.plot(x_measurment_result, color='salmon', label="noised measurement", alpha=0.5)
    plt.plot(x_lp_result, color='lime', label="low pass filter estimated", linewidth=1.5)
    plt.plot(x_kalman_result, color='blue', label="kalman filter estimated", linewidth=1.5)

    plt.legend()
    plt.show()
    