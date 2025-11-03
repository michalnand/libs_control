from kalman_filter import *
from servo_cl         import *
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


import numpy

class TrajectoryShaper:
    def __init__(self, dt, v_max, a_max, j_max, x0=0.0, damping = 1.5):
        self.dt = dt
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.damping = damping
        
        # State
        self.x = x0
        self.v = 0.0
        self.a = 0.0

    def step(self, x_desired):
        # Compute raw velocity
        #v_des = (x_desired - self.x) / self.dt

        '''
        alpha = 0.1  # tuning parameter
        v_des = alpha * (x_desired - self.x)/ self.dt

        dv = v_des - self.v
        a_raw = dv / self.dt
        da = a_raw - self.a
        j_raw = da / self.dt
        '''

        wn = numpy.sqrt(self.a_max / (self.dt * self.v_max))  # natural freq estimate
        zeta = self.damping

        a_des = wn**2 * (x_desired - self.x) - 2 * zeta * wn * self.v
        j_raw = (a_des - self.a) / self.dt

        # Limit jerk
        if abs(j_raw) > self.j_max:
            j_limited = numpy.sign(j_raw) * self.j_max
        else:
            j_limited = j_raw

        self.a += j_limited * self.dt

        # Limit acceleration
        if abs(self.a) > self.a_max:
            self.a = numpy.sign(self.a) * self.a_max

        # Integrate acceleration to get velocity
        self.v += self.a * self.dt

        # Limit velocity
        if abs(self.v) > self.v_max:
            self.v = numpy.sign(self.v) * self.v_max

        # Integrate velocity to get position
        self.x += self.v * self.dt

        return self.x




if __name__ == "__main__":
    k   = 1700
    tau = 0.8

    dt = 1.0/250.0
    
    q = numpy.diag([1.0, 0.0])
    r = numpy.diag([100.0])

  
    servo = ServoCL(k, tau, q, r, dt)

    v_max = 5000
    a_max = 10**5
    j_max = 10**10
    shaper = TrajectoryShaper(dt, v_max, a_max, j_max)


    n_steps = 1000

    t_result        = []
    x_ref_result    = []
    x_raw_result    = []

    servo.reset()
    for n in range(n_steps):

        if n < 0.1*n_steps:
            x_ref = 0.0
        elif n < 0.8*n_steps:
            x_ref = 1000.0
        else:
            x_ref = 0.0
       
        x  = servo.step(x_ref)

        t_result.append(n*dt)
        x_ref_result.append(x_ref)
        
        x_raw_result.append(x[:, 0])


    xs_result       = []
    x_smooth_result = []
    servo.reset()
    for n in range(n_steps):

        if n < 0.1*n_steps:
            xr = 0.0
        elif n < 0.8*n_steps:
            xr = 1000.0
        else:
            xr = 0.0
       
        xs = shaper.step(xr)
        x  = servo.step(xs)

        xs_result.append(xs)
        x_smooth_result.append(x[:, 0])
        

    t_result        = numpy.array(t_result)
    x_ref_result    = numpy.array(x_ref_result)
    x_raw_result    = numpy.array(x_raw_result)
    xs_result       = numpy.array(xs_result)
    x_smooth_result = numpy.array(x_smooth_result)
    
    print(">>> ", xs_result.shape, x_smooth_result.shape)
    f, t_spec, Sxx = spectrogram(x_raw_result[:, 0], 1.0/dt)
    #f, t_spec, Sxx = spectrogram(x_smooth_result[:, 0], 1.0/dt)
 

    fig, axs = plt.subplots(6, 1, figsize=(8, 8))
    axs[0].plot(t_result, x_ref_result, label="raw", color="blue", lw=2.0)
    axs[0].plot(t_result, xs_result, label="smoothed", color="red", lw=2.0)

    axs[1].plot(t_result, x_raw_result[:, 0], label="raw", color="blue", lw=2.0)
    axs[1].plot(t_result, x_smooth_result[:, 0], label="smooth", color="red", lw=2.0)
    axs[1].legend(loc="upper left")

    axs[2].plot(t_result, x_raw_result[:, 1], label="raw", color="blue", lw=2.0)
    axs[2].plot(t_result, x_smooth_result[:, 1], label="smooth", color="red", lw=2.0)
    axs[2].legend(loc="upper left")

    axs[3].plot(t_result, x_raw_result[:, 2], label="raw", color="blue", lw=2.0)
    axs[3].plot(t_result, x_smooth_result[:, 2], label="smooth", color="red", lw=2.0)
    axs[3].legend(loc="upper left")

    axs[4].plot(t_result, x_raw_result[:, 3], label="raw", color="blue", lw=2.0)
    axs[4].plot(t_result, x_smooth_result[:, 3], label="smooth", color="red", lw=2.0)
    axs[4].legend(loc="upper left")
    

    axs[5].pcolormesh(t_spec, f, 10 * numpy.log10(Sxx), shading='gouraud')

    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.show()
