import numpy as np

def steady_state_kf_gain(dt, q_j, R, tol=1e-9, max_iter=500):
    # Build F, H
    F = np.array([
        [1, dt,      dt**2/2,   dt**3/6],
        [0, 1,       dt,        dt**2/2],
        [0, 0,       1,         dt     ],
        [0, 0,       0,         1      ]
    ])
    H = np.array([[1, 0, 0, 0]])
    
    # Build Q for white jerk noise q_j
    Q = q_j * np.array([
        [   dt**7/252, dt**6/72,  dt**5/20,  dt**4/6 ],
        [   dt**6/72,  dt**5/20,  dt**4/6,   dt**3/2 ],
        [   dt**5/20,  dt**4/6,   dt**3/3,   dt**2/2 ],
        [   dt**4/6,   dt**3/2,   dt**2/2,   dt      ]
    ])
    
    # Riccati iteration
    P = np.eye(4)  # init guess
    for _ in range(max_iter):
        P_next = F@P@F.T - F@P@H.T @ np.linalg.inv(H@P@H.T + R) @ H@P@F.T + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    
    # Steady-state gain
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    return F, H, K

# Example usage:
dt  = 0.01      # 100 Hz
q_j = 1e-4      # tune to your system’s jerk variability
R   = 1e-2      # measured encoder variance

F, H, K = steady_state_kf_gain(dt, q_j, R)

# Initialize estimate
x_hat = np.zeros((4,1))

def ss_kf_update(x_hat, y_meas):
    # y_meas is scalar position measurement
    x_pred = F @ x_hat
    x_hat_new = x_pred + K @ (np.array([[y_meas]]) - H @ x_pred)
    return x_hat_new

# In your loop:
#   y = read_encoder()
#   x_hat = ss_kf_update(x_hat, y)
#   # then x_hat.flatten() gives [x, v, a, j]