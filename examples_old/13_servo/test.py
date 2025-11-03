import matplotlib.pyplot as plt
import numpy
import LibsControl

from scipy.linalg import solve_discrete_are

class Servo:
    def __init__(self, k, tau, dt = 0.001, q_vel = 0.0):

        self.dt = dt
        self.q_vel = q_vel
        # create dynamical system
        # dx = Ax + Bu
        # servo model, 2nd order
        # state x = (position, angular velocity)

        mat_a = numpy.zeros((2, 2))
        mat_b = numpy.zeros((2, 1))

        mat_a[0][1] = 1.0
        mat_a[1][1] = -1.0/tau

        mat_b[1][0] = k/tau

        #create dynamical system
        self.ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

        self.x0 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0   
        self.x3 = 0.0


    def step(self, u_in):
        self.x3 = self.x2
        self.x2 = self.x1
        self.x1 = self.x0

        u = numpy.zeros((1, 1))
        u[0, 0] = u_in
        
        x, y = self.ds.forward_state(u)


        self.x0 = x[0, 0]

        pos     = self.x0
        vel     = (self.x0 - self.x1)/self.dt
        acc     = (self.x0 - 2.0*self.x1 + self.x2)/(self.dt**2)
        jerk    = (self.x0 - 3.0*self.x1 + 3.0*self.x2 - self.x3)/(self.dt**3)


        # project noise to all states
        std_vel = numpy.sqrt(self.q_vel)

        pos_n   = pos  + (std_vel*self.dt)*numpy.random.randn()
        vel_n   = vel  + (std_vel)*numpy.random.randn()
        acc_n   = acc  + (std_vel/self.dt)*numpy.random.randn()
        jerk_n  = jerk + (std_vel/(self.dt**2))*numpy.random.randn()


        x        = numpy.array([[pos, vel, acc, jerk]]).T
        x_noised = numpy.array([[pos_n, vel_n, acc_n, jerk_n]]).T

        return x, x_noised



class KalmanFilter:

    def __init__(self, A, H, Q, R):
        self.A = numpy.array(A)
        self.H = numpy.array(H)

        # Solve Discrete Algebraic Riccati Equation (DARE)
        P = solve_discrete_are(A.T, H.T, Q, R)

        # Compute steady-state Kalman Gain
        self.K = P @ H.T @ numpy.linalg.inv(H @ P @ H.T + R)

        print(self.A)
        print(self.H)
        print(self.K)


    def step(self, y, x_hat):
        x_pred    = self.A@x_hat
        x_hat_new = x_pred + self.K @ (y - self.H @ x_pred)

        return x_hat_new



if __name__ == "__main__":
    k   = 1700
    tau = 0.2

    dt = 1.0/250.0

    q_vel = 1

    servo = Servo(k, tau, dt, q_vel)

    # State Transition Matrix
    '''
    a = numpy.array([
        [1, dt, 0.5*dt**2, (1/6)*dt**3],
        [0, 1, dt, 0.5*dt**2],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    '''

    a = numpy.array([
        [1, dt, 0, 0],
        [0, 1, dt, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    
    h = numpy.array([[1, 0, 0, 0]])

    # Project noise to all states
    q_pos  = q_vel * dt            # [rad^2]
    q_acc  = 1000*q_vel / dt            # [rad^2/s^4]
    q_jerk = 1000000*q_vel / (dt**2)      # [rad^2/s^6]
    
    # Build Process Noise Covariance Matrix
    q = numpy.diag([q_pos, q_vel, q_acc, q_jerk])

    # Measurement noise covariance
    r_pos = 1e-100
    r = numpy.array([[r_pos]])

    kf = KalmanFilter(a, h, q, r)

    x_hat = numpy.zeros((a.shape[0], 1))


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

        x_pos = numpy.array([[x_noised[0][0]]])

        x_hat = kf.step(x_pos, x_hat)

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


