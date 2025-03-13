import numpy
import numpy as np


def hankel_matrix(data, L):
    """
    Constructs a block Hankel matrix from data.
    
    Parameters:
      data: array of shape (n_samples, n_features)
      L: number of block rows (horizon)
      
    Returns:
      H: block Hankel matrix of shape (L*n_features, n_columns)
         where n_columns = n_samples - L + 1.
    """
    n_samples, n_features = data.shape
    n_columns = n_samples - L + 1
    H = np.zeros((L * n_features, n_columns))
    for i in range(L):
        # Each block row is the transposed slice of data.
        H[i*n_features:(i+1)*n_features, :] = data[i:i+n_columns].T
    return H


def moesp(X, U, L):
    """
    MOESP: Estimate system matrices A and B using QR decomposition for the projection.
    
    Parameters:
      X: state sequence, shape (n_samples, n_states)
      U: input sequence, shape (n_samples, n_inputs)
      L: horizon (number of block rows)
      
    Returns:
      A_est: Estimated state transition matrix A.
      B_est: Estimated input matrix B.
    """
    # Use consistent time indices for both X and U.
    # Future state data: one-step ahead of X.
    X_f = hankel_matrix(X[1:], L)      # Uses X[1:] so n_columns = (n_samples-1) - L + 1
    # Past input data: use U[:-1] to match the time span of X[1:].
    U_p = hankel_matrix(U[:-1], L)     # Now n_columns is the same as X_f
    
    n_state = X.shape[1]
    
    # --- QR-based Projection ---
    # We wish to project onto the orthogonal complement of U_p's row space.
    # Compute the complete QR decomposition of U_p.T.
    Q_full, R_full = np.linalg.qr(U_p.T, mode='complete')
    # U_p has shape (L*n_inputs, n_columns) so U_p.T is (n_columns, L*n_inputs).
    # Let r be the rank (ideally L*n_inputs). Then Q_full's first r columns span U_p.T's range.
    r = U_p.shape[0]  # This assumes full row rank.
    Q2 = Q_full[:, r:]  # Q2 spans the orthogonal complement.
    # Projection matrix onto the null space of U_p.
    P_perp = Q2 @ Q2.T              # shape: (n_columns, n_columns)
    
    # Project the future state Hankel matrix.
    X_f_proj = X_f @ P_perp         # Now inner dimensions match.
    
    # SVD of the projected data.
    U_svd, Sigma, Vh = np.linalg.svd(X_f_proj, full_matrices=False)
    U1 = U_svd[:, :n_state]
    Sigma1 = np.diag(Sigma[:n_state])
    Gamma = U1 @ np.sqrt(Sigma1)      # Extended observability matrix
    
    # Partition Gamma into block rows.
    Gamma_reshaped = Gamma.reshape(L, X.shape[1], n_state)
    Gamma_upper = Gamma_reshaped[:-1].reshape((L-1)*X.shape[1], n_state)
    Gamma_lower = Gamma_reshaped[1:].reshape((L-1)*X.shape[1], n_state)
    
    # Recover A using the shift invariance property.
    A_est = np.linalg.lstsq(Gamma_upper, Gamma_lower, rcond=None)[0]
    
    # Estimate B via one-step regression from the original data.
    X_future = X[1:]   # shape: (n_samples-1, n_states)
    X_past   = X[:-1]  # shape: (n_samples-1, n_states)
    U_past   = U[:-1]  # shape: (n_samples-1, n_inputs)
    regressor = np.hstack((X_past, U_past))
    AB = np.linalg.lstsq(regressor, X_future, rcond=None)[0]
    AB = AB.T
    # Split the estimated [A B] matrix.
    B_est = AB[:, X.shape[1]:]
    
    return A_est, B_est



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

    a_est, b_est = moesp(x_in, u, 2)

    print("\n\n")
    print(a_est)
    print(b_est)

    error_a = ((mat_a - a_est)**2).mean()
    error_b = ((mat_b - b_est)**2).mean()

    print(error_a, error_b)
