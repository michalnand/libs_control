import numpy




'''
recursive least squares system identification 
this algorithm can run real time, 
doesn't requires matrix inversion, 
or long term memory for data

finds matrices A, B for discrete model :

    x(n+1) = Ax(n) + Bu(n)

returns model matrices A_hat, B_hat

inputs : 
    u : controll input shape(time_steps, n_inputs)
    x : state shape(time_steps, n_states)

returns :
    A : shape (n_states, n_states)
    B : shape (n_states, n_inputs)
'''
def rls_identification(u, x, a_initial = None, b_initial = None):    
    n_samples = x.shape[0]
    n_states  = x.shape[1]
    n_inputs  = u.shape[1]

    # model variable
    theta = numpy.zeros((n_states, n_states + n_inputs))

    if a_initial is not None:
        theta[:, 0:n_states] = a_initial.copy()
    
    if b_initial is not None:
        theta[:, n_states:] = b_initial.copy()

    #initial covariance
    P     = numpy.eye(n_states + n_inputs) 

    lam   = 1e-6

    for n in range(1, n_samples):
        # prepare inputs
        x_prev = numpy.expand_dims(x[n-1], 1)
        x_now  = numpy.expand_dims(x[n], 1)
        u_prev = numpy.expand_dims(u[n-1], 1)

        # augmented inputs matrix
        xu = numpy.concatenate((x_prev, u_prev), axis=0)
        
        # model prediction error
        error = x_now - theta@xu
    
        # RLS update
        denom = (lam + xu.T @ P @ xu).item()

        if numpy.abs(denom) > 10e-8 and numpy.abs(denom) < 10e8:
            #kalman gain
            K   = (P @ xu)/denom
            
            theta = theta + error @ K.T
            P     = (numpy.eye(n_states + n_inputs) - K @ xu.T) @ P


    a_est = theta[:, 0:n_states]
    b_est = theta[:, n_states:]

    return a_est, b_est



'''
    Q - process noise,     shape = (n_states, n_states)
    R - observation noise, shape = (n_states, n_states)
    alpha - 0..1, adaptive R update rate, 1 == to no update
'''
def _kalman_filter(u, x, A, B, Q, R, alpha = 0.9):
    n_samples = x.shape[0]
    n_states  = x.shape[1]

    x_pred = numpy.zeros((n_samples, n_states))
    x_est  = numpy.zeros((n_samples, n_states))
    P      = numpy.eye(n_states)

    x_est[0, :] = x[0, :]

    for t in range(1, n_samples):
       
        # prediction
        x_pred[t, :] = A @ x_est[t-1, :] + B @ u[t-1, :]
        P = A @ P @ A.T + Q

        error = x[t, :] - x_pred[t, :]

        # correction
        K = P @ numpy.eye(n_states).T @ numpy.linalg.inv(numpy.eye(n_states) @ P @ numpy.eye(n_states).T + R)
        
        x_est[t, :] = x_pred[t, :] + K @ (error)
        
        P = (numpy.eye(n_states) - K @ numpy.eye(n_states)) @ P
        
        # adaptive observation noise estimation
        R = alpha * R + (1 - alpha) * (error.reshape(-1, 1) @ error.reshape(1, -1))

   
    return x_est


'''
Expectation-Maximization algorithm :
    for n in num_iterations:
        x_est = kalman_filter(u, x, a_est, b_est, q, r)
        a_est, b_est = rls_identification(u, x_est, a_initial = a_est, b_initial = b_est)  
'''
def krls_identification(u, x, num_iterations = 10, alpha = 0.95):    
    n_states  = x.shape[1]

   
    q = (10**-20)*numpy.eye(n_states)
    r = 0.1*numpy.eye(n_states)
    
    #initial estimation
    a_est, b_est = None, None

    x_est = x.copy()
    # iterations for EM
    for _ in range(num_iterations):  
        # maximization step
        a_est, b_est = rls_identification(u, x_est, a_initial = a_est, b_initial = b_est)
        
        # expectation step
        x_est = _kalman_filter(u, x, a_est, b_est, q, r, alpha)

    
    return a_est, b_est


