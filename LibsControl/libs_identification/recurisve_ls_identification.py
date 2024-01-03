import numpy

'''
recursive least squares system identification for discrete model :

    x(n+1) = Ax(n) + Bu(n)

returns model matrices A_hat, B_hat

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