from kalman_filter import *
from servo         import *
import matplotlib.pyplot as plt

def compute_A(dt):
    return numpy.array([
        [1, dt, dt**2 / 2, dt**3 / 6],
        [0, 1, dt, dt**2 / 2],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

def compute_Q(sigma_j2, dt):
    Q = sigma_j2 * numpy.array([
        [dt**6 / 36, dt**5 / 12, dt**4 / 6, dt**3 / 6],
        [dt**5 / 12, dt**4 / 4, dt**3 / 2, dt**2 / 2],
        [dt**4 / 6, dt**3 / 2, dt**2, dt],
        [dt**3 / 6, dt**2 / 2, dt, 1]
    ])
    return Q

def compute_R(sigma_v2, dt):
    R = numpy.diag([
        0.5 * dt**2 * sigma_v2,   # position
        sigma_v2,                 # velocity
        3.0 * sigma_v2 / dt**2,   # acceleration
        10.0 * sigma_v2 / dt**4   # jerk
    ])
    return R

if __name__ == "__main__":
    k   = 1700
    tau = 0.2

    dt = 1.0/250.0
    
    sigma_v2 = 10.0
    sigma_j2 = 10**10

    A = compute_A(dt)
    H = numpy.eye(A.shape[0])
    R = compute_R(sigma_v2, dt)
    Q = compute_Q(sigma_j2, dt)

    print(R)
    print(Q)
    



    servo = Servo(k, tau, dt, R)


    kf      = KalmanFilter(A, H, Q, R)
    x_hat   = numpy.zeros((A.shape[0], 1))

    n_steps = 1000
    

    t_result        = []
    u_result        = []
    x_result        = []
    x_noised_result = []

    x_hat_result    = []

    for n in range(n_steps):

        if n < 0.1*n_steps:
            u = 0.0
        elif n < 0.3*n_steps:
            u = 1.0
        elif n < 0.7*n_steps:
            u = 0.0
        elif n < 0.9*n_steps:
            u = -1.0
        elif n < 1.0*n_steps:
            u = 0.0

        x, x_noised = servo.step(u)

        x_hat = kf.step(x_noised, x_hat)

        t_result.append(n*dt)
        u_result.append(u)
        x_result.append(x[:, 0])
        x_noised_result.append(x_noised[:, 0])
        x_hat_result.append(x_hat[:, 0])


    t_result        = numpy.array(t_result)
    u_result        = numpy.array(u_result)
    x_result        = numpy.array(x_result)
    x_noised_result = numpy.array(x_noised_result)
    x_hat_result    = numpy.array(x_hat_result)


    print(x_result.shape)
    print(x_noised_result.shape)


    fig, axs = plt.subplots(5, 1, figsize=(8, 8))
    axs[0].plot(t_result, u_result)

    axs[1].plot(t_result, x_result[:, 0], label="gt", color="blue", lw=4.0)
    axs[1].plot(t_result, x_noised_result[:, 0], label="noised", color="purple", alpha=0.5)
    axs[1].plot(t_result, x_hat_result[:, 0], label="estimated", color="red", alpha=1.0, lw=2.0)
    axs[1].legend(loc="upper left")

    axs[2].plot(t_result, x_result[:, 1], label="gt", color="blue", lw=4.0)
    axs[2].plot(t_result, x_noised_result[:, 1], label="noised", color="purple", alpha=0.5)
    axs[2].plot(t_result, x_hat_result[:, 1], label="estimated", color="red", alpha=1.0, lw=2.0)
    axs[2].legend(loc="upper left")

    axs[3].plot(t_result, x_result[:, 2], label="gt", color="blue", lw=4.0)
    axs[3].plot(t_result, x_noised_result[:, 2], label="noised", color="purple", alpha=0.5)
    axs[3].plot(t_result, x_hat_result[:, 2], label="estimated", color="red", alpha=1.0, lw=2.0)
    axs[3].legend(loc="upper left")

    axs[4].plot(t_result, x_result[:, 3], label="gt", color="blue", lw=4.0)
    axs[4].plot(t_result, x_noised_result[:, 3], label="noised", color="purple", alpha=0.5)
    axs[4].plot(t_result, x_hat_result[:, 3], label="estimated", color="red", alpha=1.0, lw=2.0)
    axs[4].legend(loc="upper left")
    

    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.show()
