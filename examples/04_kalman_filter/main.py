import numpy
import matplotlib.pyplot as plt
import LibsControl

from kalman_filter import *

class EncoderModel:
    def __init__(self, velocity_noise_var,  q_var, dt):
        # state x = [pos; vel]
        self.x = numpy.zeros((2, 1))

        # State transition matrix
        self.A = numpy.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # noise standard deviations
        self.vel_std = numpy.sqrt(velocity_noise_var)
        self.pos_std = 0.5*self.vel_std


        self.R = numpy.zeros((2, 2))
        self.R[0][0] = self.pos_std
        self.R[1][1] = self.vel_std

        self.Q = numpy.sqrt(q_var)*numpy.eye(2)

    def step(self, u):
        x_true = self.A @ self.x
        x_true[1, 0] = u

        x_obs = x_true.copy()
        x_obs[0, 0] += numpy.random.randn() * self.pos_std
        x_obs[1, 0] += numpy.random.randn() * self.vel_std

        self.x = x_true

        return x_true, x_obs


def random_u(n_steps, p = 0.02):
    u_res = []

    u = 0.0
    for n in range(n_steps):
        if numpy.random.rand() < p:
            u = numpy.random.randn()

        u_res.append(u)

    u_res = numpy.array(u_res)
    return numpy.expand_dims(u_res, 1)

if __name__ == "__main__":
    
    n_steps = 1000

    velocity_noise_var = 0.25
    dt = 1.0/100.0

    u_result = random_u(n_steps)

    print(u_result.shape)

    encoder = EncoderModel(velocity_noise_var, 0.0001, dt)

    kf = KalmanFilter(encoder.A, encoder.Q, encoder.R)

    print(kf.K)

    t_result = []

    x_true = []
    x_obs  = []

    x_hat = []

    for n in range(n_steps):

        x_true_, x_obs_ = encoder.step(u_result[n, 0])

        x_hat_ = kf.step(x_obs_)

        t_result.append(n*dt)

        x_true.append(x_true_[:, 0])
        x_obs.append(x_obs_[:, 0])
        x_hat.append(x_hat_[:, 0])

    x_true = numpy.array(x_true)
    x_obs  = numpy.array(x_obs)
    x_hat  = numpy.array(x_hat)


    #LibsControl.plot_cl_response(t_result, u_result, x_true, x_obs, "encoder.png",  ["input"],  ["position", "velocity"])
    LibsControl.plot_cl_response(t_result, u_result, x_obs, x_hat, "encoder.png",  ["input"],  ["position", "velocity"])
