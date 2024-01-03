import numpy

'''
def kalman_observer_identification(u, x, lr = 0.001):

    noise_std_state = 0.1
    noise_std_input = 0.001

    # Kalman filter initialization for 1D system
    estimated_a = 1.0  # Initial guess for a
    estimated_b = 1.0  # Initial guess for b
    P_ident = noise_std_state**2  # Initial state estimation covariance for identification
    Q_ident = noise_std_state**2  # Process noise covariance for identification
    R_ident = noise_std_input**2  # Measurement noise covariance for identification

    # Kalman filter-based identification for 1D system
    num_samples = u.shape[0]
    for k in range(1, num_samples):

        x_prev = x[k-1][0]
        x_now  = x[k][0]
        u_prev = u[k-1][0]

        # Prediction step using estimated parameters
        x_pred = estimated_a * x_prev + estimated_b * u_prev
        P_pred = estimated_a**2 * P_ident + Q_ident
        
        # Kalman gain calculation
        K = P_pred / (P_pred + R_ident)

        #K = numpy.clip(K, -1.0, 1.0)

        print(K, estimated_a, estimated_b)
        
        # Update step
        x_residual = x_now - x_pred

        estimated_a += lr*K * x_residual * x_prev
        estimated_b += lr*K * x_residual * u_prev
        P_ident = (1 - K) * P_pred

        #print(x_residual, estimated_a, estimated_b)

    return estimated_a, estimated_b
'''


'''
def kalman_observer_identification(u, x, n_states, n_inputs, noise_std_state = 0.1, noise_std_input = 0.1, lr = 0.01):
    #kalman filter initialization
    estimated_a = numpy.eye(n_states)
    estimated_b = numpy.zeros((n_states, n_inputs))

    #state estimation covariance
    P_ident = (noise_std_state**2)*numpy.eye(n_states)

    #process noise covariance
    Q_ident = (noise_std_state**2)*numpy.eye(n_states)

    #measurement noise covariance
    R_ident = (noise_std_input**2)*numpy.eye(n_inputs)

    #kalman filter-based identification
    num_samples = u.shape[0]
    for n in range(1, num_samples):
        #obtain inputs
        x_prev = numpy.expand_dims(x[n-1], 1)
        x_now  = numpy.expand_dims(x[n], 1)
        u_prev = numpy.expand_dims(u[n-1], 1)

        #prediction step using estimated parameters
        x_pred = estimated_a@x_prev + estimated_b@u_prev
        P_pred = estimated_a@P_ident@estimated_a.T + Q_ident


        #kalman gain calculation
        K = P_pred / (P_pred + R_ident)

        #print(n)
        #print(P_pred)
        #print("\n\n")
 
        #update step
        x_residual = x_now - x_pred
        P_ident = (1.0 - K)@P_pred

        #P_ident = numpy.maximum(P_ident, 0.1*numpy.eye(n_states))

        P_ident = numpy.clip(P_ident, 0.01, 10**2)

        print(P_ident)
        print("\n\n\n")
        #parameter estimation
        estimated_a += lr*K@x_residual@x_prev.T
        estimated_b += lr*K@x_residual@u_prev.T
        
    return estimated_a, estimated_b
'''


'''
recursive least squares system identification for model

x(n+1) = Ax(n) + Bu(n)

returns matrices A, B

inputs : 
u : controll input shape(time_steps, n_inputs)
x : state shape(time_steps, n_states)

returns :
A : shape (n_states, n_states)
B : shape (n_states, n_inputs)
'''
def recurisve_ls_identification(u, x):
    # resulted model parameters
    n_states = x.shape[1]
    n_inputs = u.shape[1]

    theta = numpy.zeros((n_states, n_states + n_inputs))

    # intial P covariance matrix
    P = numpy.eye(n_states + n_inputs) 

    # forgetting factor
    lambda_val = 0.99 

    num_samples = u.shape[0]
    for n in range(1, num_samples):
        #obtain inputs
        x_prev = numpy.expand_dims(x[n-1], 1)
        x_now  = numpy.expand_dims(x[n], 1)
        u_prev = numpy.expand_dims(u[n-1], 1)

        # augmented inputs matrix
        extended_x = numpy.concatenate((x_prev, u_prev), axis=0)
        
        # model prediction error
        error = x_now - theta@extended_x

        # Kalman gain    
        denom = (lambda_val + extended_x.T@P@extended_x)[0][0]
        if numpy.abs(denom) > 10e-4 and numpy.abs(denom) < 10e3:
            K = (P@extended_x) / denom
            # model update
            theta += (error@K.T)
            # covariance update
            P = (1.0 / lambda_val) * (P - K@extended_x.T@P)

    a_est = theta[:, 0:n_states]
    b_est = theta[:, n_states:]
    return a_est, b_est

if __name__ == "__main__":

    n_states = 6
    n_inputs = 2

    
    a_ref = 0.1*numpy.random.randn(n_states, n_states)
    b_ref = numpy.random.randn(n_states, n_inputs) 

    a_ref[range(0, n_states, 2), range(0, n_states, 2)] = 1.0

    print("true system : \n")
    print(numpy.round(a_ref, 4))
    print(numpy.round(b_ref, 4))
    print("\n\n")

    x = numpy.random.randn(n_states, 1)
    x_prev = x.copy()


    
    x_log = []
    u_log = []
    for n in range(10000):
        u      = numpy.random.randn(n_inputs, 1)
        x_next = a_ref@x + b_ref@u

        x_log.append(x[:, 0])
        u_log.append(u[:, 0])

        x = x_next.copy()


    x_log = numpy.array(x_log)
    u_log = numpy.array(u_log)

    x_log+= 0.1*numpy.random.randn(x_log.shape[0], x_log.shape[1])


    #a_hat, b_hat = kalman_observer_identification(u_log, x_log, n_states, n_inputs)

    a_hat, b_hat = recurisve_ls_identification(u_log, x_log)



    print(numpy.round(a_hat, 4))
    print(numpy.round(b_hat, 4))

    

    a_dif = numpy.abs(a_ref - a_hat)
    b_dif = numpy.abs(b_ref - b_hat)

    a_rel = (a_dif/numpy.abs(a_ref)).mean()
    b_rel = (b_dif/numpy.abs(b_ref)).mean()


    print("a_mse = ", a_dif.mean(), a_rel)
    print("b_mse = ", b_dif.mean(), b_rel)  


    ref_poles = numpy.linalg.eigvals(a_ref)
    hat_poles = numpy.linalg.eigvals(a_hat)

    print(ref_poles)
    print(hat_poles)