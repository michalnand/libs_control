import numpy

def construct_prediction_matrix(x, u, A_init, B_init):
    N = x.shape[0] - 1  # number of data points (one less because we predict x(n+1) from x(n))
    n = x.shape[1]      # state dimension
    m = u.shape[1]      # input dimension

    P = []
    X_pred = []

    for i in range(N):
        row_A = numpy.zeros((n, n * n))
        row_B = numpy.zeros((n, n * m))


        for j in range(i + 1):
            A_power = numpy.linalg.matrix_power(A_init, i - j)
            row_A += numpy.kron(numpy.eye(n), A_power)
            row_B += numpy.kron(numpy.eye(n), A_power @ B_init)
        

        P.append(numpy.hstack((row_A, row_B)))
        X_pred.append(x[i + 1].flatten())

    P = numpy.vstack(P)
    X_pred = numpy.hstack(X_pred)

    return P, X_pred

def system_identification(x, u, A_init, B_init):
    P, X_pred = construct_prediction_matrix(x, u, A_init, B_init)
    
    # Solve the least squares problem
    theta, _, _, _ = numpy.linalg.lstsq(P, X_pred, rcond=None)
    
    n = A_init.shape[0]
    m = B_init.shape[1]
    
    A_estimated = theta[:n*n].reshape(n, n)
    B_estimated = theta[n*n:].reshape(n, m)
    
    return A_estimated, B_estimated

# Example usage
n_steps   = 100
state_dim = 5  # Example state dimension
input_dim = 3  # Example input dimension    

x = numpy.random.rand(n_steps, state_dim)  # Observed state sequence (n_steps time steps)
u = numpy.random.rand(n_steps, input_dim)  # Input sequence (n_steps time steps)

A_init = numpy.random.rand(state_dim, state_dim)  # Initial guess for A
B_init = numpy.random.rand(state_dim, input_dim)  # Initial guess for B

A_estimated, B_estimated = system_identification(x, u, A_init, B_init)
print("Estimated A:", A_estimated)
print("Estimated B:", B_estimated)