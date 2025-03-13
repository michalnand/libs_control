import numpy
import numpy as np




def qr_system_identification(x, u):

    n_x = x.shape[1]

    # --- Construct Data Matrices ---
    # Use the first n_samples-1 rows for X and U, and rows 1 to n_samples for X_next.
    X = x[:-1, :]       # shape: (n_samples-1, n_x)
    X_next = x[1:, :]   # shape: (n_samples-1, n_x)
    U = u[:-1, :]       # shape: (n_samples-1, n_u)

    # Form the regression matrix by concatenating state and input measurements horizontally.
    Z = numpy.hstack([X, U])  # shape: (n_samples-1, n_x+n_u)

    # --- QR Decomposition ---
    # Compute the reduced QR decomposition of Z.
    # Here, Z = Q * R where Q has shape (n_samples-1, n_x+n_u) and R is square.
    Q, R = numpy.linalg.qr(Z, mode='reduced')

    # --- Least-Squares Estimate ---
    # Solve for Theta in the regression X_next = Z * Theta.
    # The least-squares solution is Theta = R^{-1} Q^T * X_next.
    Theta = numpy.linalg.inv(R).dot(Q.T).dot(X_next)  # Theta has shape (n_x+n_u, n_x)

    # Partition Theta into estimates for A^T and B^T, then transpose to get A and B.
    A_est = Theta[:n_x, :].T  # Estimated A (shape: n_x x n_x)
    B_est = Theta[n_x:, :].T  # Estimated B (shape: n_x x n_u)

    return A_est, B_est







def construct_hankel(data, p, L):
    """
    Constructs a block Hankel matrix with the intended shape (p*d, L).

    Parameters:
      data: array of shape (N, d) [each row is one time sample]
      p: number of block rows
      L: number of columns in the Hankel matrix (L = N - p - offset)

    Returns:
      H: block Hankel matrix of shape (p*d, L)
    """
    # Each block: data[i:i+L] has shape (L, d). Transpose to (d, L) and then stack.
    H = np.vstack([data[i:i+L].T for i in range(p)])
    return H

def identify_AB(x, u, p):
    """
    Identify state-space matrices A and B from measured state and input data,
    using block Hankel matrices and a QR-based pseudo-inverse.

    Parameters:
      x: state trajectory, shape (N, n_states)
      u: input trajectory, shape (N, n_inputs)
      p: number of block rows to use in Hankel matrices (choose p < N-1)

    Returns:
      A: state matrix of shape (n_states, n_states)
      B: input matrix of shape (n_states, n_inputs)
    """
    N, n_states = x.shape
    N_u, n_inputs = u.shape
    assert N == N_u, "State and input must have the same number of samples."

    # Choose L = N - p - 1 so that the future-shifted Hankel has the same column count.
    L = N - p - 1  

    # Construct past Hankel matrices:
    # Hx will have shape (p*n_states, L) and Hu will have shape (p*n_inputs, L)
    Hx = construct_hankel(x, p, L)        
    Hu = construct_hankel(u, p, L)        

    # Construct future state Hankel matrix (shift x by one time step)
    # x[1:] has shape (N-1, n_states) so Hxf will have shape (p*n_states, L)
    Hxf = construct_hankel(x[1:], p, L)     

    # Form the combined past data matrix M:
    # M has shape ((p*n_states + p*n_inputs), L)
    M = np.vstack([Hx, Hu])
    
    # --- QR Decomposition Approach ---
    # Compute the QR decomposition of M^T:  M^T = Q * R.
    # M has shape (m, L) with m = p*(n_states+n_inputs). So M^T has shape (L, m).
    Q, R = np.linalg.qr(M.T, mode="reduced")   # Q: (L, m), R: (m, m)

    err= ((Q@R - M.T)**2).mean()
    print("QR = ", err, M.shape, Hx.shape, Hu.shape)

    # Compute the pseudo-inverse of M using the QR factors:
    # For M of shape (m, L) with m < L, a proper pseudo-inverse is:
    # M_pinv = Q @ inv(R.T) which yields shape (L, m)
    M_pinv = Q @ np.linalg.inv(R.T)

    #reg = 1e-12
    #M_pinv = Q @ np.linalg.pinv(R.T + reg*np.eye(R.shape[0]))

    # Extended least squares: solve Hxf = [A_ext] * M  ==>  [A_ext] = Hxf @ M_pinv.
    # Hxf: shape (p*n_states, L), M_pinv: shape (L, p*(n_states+n_inputs))
    AB_extended = Hxf @ M_pinv           

    # Extract the first block row corresponding to the immediate one-step dynamics.
    # That is, take the first n_states rows and the first (n_states+n_inputs) columns.
    AB = AB_extended[:n_states, : (n_states + n_inputs)]  

    # Separate into A and B.
    A = AB[:, :n_states]    # shape: (n_states, n_states)
    B = AB[:, n_states:]    # shape: (n_states, n_inputs)
    
    return A, B

if __name__ == '__main__':

    n_samples   = 200

    n_inputs    = 3
    n_order     = 5

    mat_a = 0.5*numpy.random.randn(n_order, n_order)
    mat_b = 0.5*numpy.random.randn(n_order, n_inputs)

    u = numpy.random.randn(n_samples, n_inputs)

    x = numpy.zeros((n_samples,n_order))

    for n in range(n_samples-1):
        x_curr= numpy.expand_dims(x[n], 1)
        u_curr= numpy.expand_dims(u[n], 1)
        x_new = mat_a@x_curr + mat_b@u_curr

        x[n+1] = x_new[:, 0]


    print(mat_a)
    print(mat_b)

    x_in = x #+ numpy.random.randn(n_samples, n_order)

    a_est, b_est = identify_AB(x_in, u, 2*n_order)

    print("\n\n")
    print(a_est)
    print(b_est)

    error_a = ((mat_a - a_est)**2).mean()
    error_b = ((mat_b - b_est)**2).mean()

    print(error_a, error_b)
