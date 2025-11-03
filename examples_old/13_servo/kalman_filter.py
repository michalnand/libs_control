import numpy
from scipy.linalg import solve_discrete_are


class KalmanFilter:

    def __init__(self, A, H, Q, R):
        self.A = numpy.array(A)
        self.H = numpy.array(H)

        # Solve Discrete Algebraic Riccati Equation (DARE)
        #P = solve_discrete_are(A.T, H.T, Q, R)

        # Compute steady-state Kalman Gain
        #self.K = P @ H.T @ numpy.linalg.inv(H @ P @ H.T + R)

        P = solve_discrete_are(A.T, H.T, Q, R)
        self.K = P @ H.T @ numpy.linalg.inv(H @ P @ H.T + R)

        print(self.A)
        print(self.H)
        print(self.K)


    def step(self, y, x_hat):
        x_pred    = self.A@x_hat
        x_hat_new = x_pred + self.K @ (y - self.H @ x_pred)

        return x_hat_new