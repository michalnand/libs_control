import numpy
import matplotlib.pyplot as plt


def ls_identification(u, x):
    x_now  = x[0:-1, :]
    u_now  = u[0:-1, :]

    x_next = x[1:, :]

    x_tmp = numpy.hstack([x_now, u_now])

    theta = numpy.linalg.lstsq(x_tmp, x_next, rcond=None)[0]

    a = theta[0:x_now.shape[1], :]
    b = theta[x_now.shape[1]: , :]

    a = numpy.array(a.T)
    b = numpy.array(b.T)

    return a, b


def qr_identification(u, x):
    x_now  = x[0:-1, :]
    u_now  = u[0:-1, :]

    x_next = x[1:, :]

    x_tmp = numpy.hstack([x_now, u_now])

    # QR decomposition of input
    Q, R = numpy.linalg.qr(x_tmp)
    
    # Compute Q^T y
    Qt_y = Q.T @ x_next

    # Solve RM = Q^T y for M
    # Use numpy.linalg.solve for solving R @ M = Qt_y
    theta = numpy.linalg.solve(R, Qt_y)
    

    a = theta[0:x_now.shape[1], :]
    b = theta[x_now.shape[1]: , :]

    a = numpy.array(a.T)
    b = numpy.array(b.T)

    return a, b



def subspace_id(U, X, h, sys_order=None, refine_B=True):
    """
    MOESP subspace-ID with data centering, order selection, and B‑refinement.

    Args:
      X          (N×n) state snapshots
      U          (N×m) input snapshots
      h          block horizon (2n ≳ h < N/2)
      sys_order  if set, truncates SVD to this order
      refine_B   if True, re‑estimate B via one‑step LS after A

    Returns:
      A_hat (n×n), B_hat (n×m)
    """
    # 0) center data
    X = X - X.mean(axis=0, keepdims=True)
    U = U - U.mean(axis=0, keepdims=True)

    N, n = X.shape
    _, m = U.shape
    K = N - 2*h + 1
    if K <= 0:
        raise ValueError("Need N > 2h-1.")

    # 1) block‐Hankel
    def hankel(data):
        d = data.shape[1]
        H = numpy.zeros((h*d, K))
        for i in range(h):
            H[i*d:(i+1)*d, :] = data[i:i+K, :].T
        return H

    Up = hankel(U)         # (h*m × K)
    Xp = hankel(X)         # (h*n × K)
    Xf = hankel(X[h:,:])   # (h*n × K)

    # 2) project future onto past
    Wp = numpy.vstack([Up, Xp])         # (h*(m+n) × K)
    P  = Xf @ numpy.linalg.pinv(Wp)     # (h*n × h*(m+n))

    # 3) SVD → extended observability
    U_s, S_s, _ = numpy.linalg.svd(P, full_matrices=False)
    r = sys_order if sys_order is not None else n
    Gamma = U_s[:, :r] @ numpy.diag(numpy.sqrt(S_s[:r]))  # (h*n × r)

    # 4) estimate A
    Gam_up  = Gamma[:-n, :]  # ((h-1)*n × r)
    Gam_low = Gamma[ n:, :]  # ((h-1)*n × r)
    A_hat = numpy.linalg.lstsq(Gam_up, Gam_low, rcond=None)[0]

    # 5) initial B from multi-step (optional, not used if refine_B)
    #  Xp1 = Xp[0:n, :]      # n×K
    #  Hh  = (Xf - Gamma @ Xp1) @ numpy.linalg.pinv(Up)
    #  B0  = Hh[:n, :m]

    # 6) refine B via one-step LS
    if refine_B:
        X1 = X[1:].T                         # n × (N-1)
        H1 = numpy.vstack([A_hat @ X[:-1].T,    # n × (N-1)
                        U[:-1].T])           # m × (N-1)
        Theta = X1 @ numpy.linalg.pinv(H1)      # n × (n+m)
        B_hat = Theta[:, n:]
    else:
        Xp1 = Xp[0:n, :]
        Hh  = (Xf - Gamma @ Xp1) @ numpy.linalg.pinv(Up)
        B_hat = Hh[:n, :m]

    return A_hat, B_hat



def eval(a_gt, b_gt, a_hat, b_hat):

    x_gt  = numpy.hstack([a_gt, b_gt])
    x_hat = numpy.hstack([a_hat, b_hat])

    diff = numpy.abs(x_gt - x_hat)


    mse     = (diff**2).mean()
    mape    = 100.0*(diff/(numpy.abs(x_gt) + 10e-6)).mean()

    return mse, mape


def random_stable(system_order):

    r = 1.0
    while True:
        a   = r*numpy.random.randn(system_order, system_order)
        eig = numpy.linalg.eigvals(a)

        mag = numpy.abs(eig)

        if numpy.max(mag) <= 1.0:
            break
        
        r = r*0.99
    
    return a, eig


def get_response(a, b, u):
    u_result = []
    x_result = []

    x_now = numpy.zeros((system_order, 1))

    for n in range(n_samples):
        u_now = numpy.zeros((system_inputs, 1))
        u_now[:, 0] = u[n]

        x_result.append(x_now[:, 0])
        u_result.append(u_now[:, 0])

        x_next = a@x_now + b@u_now

        x_now = numpy.array(x_next)

    u_result = numpy.array(u_result)
    x_result = numpy.array(x_result)

    return x_result
 


if __name__ == "__main__":
    system_order    = 17
    system_inputs   = 7

    n_samples       = 500


    n_levels    = 10
    n_systems   = 10

    n_methods       = 3
    noise_levels    = numpy.zeros((n_levels, ))
    result_mse      = numpy.zeros((n_methods, n_levels, n_systems))
    result_mape     = numpy.zeros((n_methods, n_levels, n_systems))

    eig_vals = []

    for k in range(n_levels):
        noise_level = k/(n_levels-1)

        noise_levels[k] = noise_level

        for n in range(n_systems):
            # create random dynamical system
            a_gt, eig = random_stable(system_order)
            b_gt = numpy.random.randn(system_order, system_inputs) 

            eig_vals.append(eig)

            # obtaine response
            u = numpy.random.randn(n_samples, system_inputs)
            x = get_response(a_gt, b_gt, u)

            # add noise            
            x_noised = x + noise_level*numpy.random.randn(x.shape[0], x.shape[1])

            # obtain GT response
            u_ref = u[0:100, :]
            x_ref = get_response(a_gt, b_gt, u_ref)

            # system identification

            # least squares
            a_ls, b_ls = ls_identification(u, x_noised)
            mse, mape = eval(a_gt, b_gt, a_ls, b_ls)

            result_mse[0][k][n]  = mse
            result_mape[0][k][n] = mape


            a_mls, b_mls = qr_identification(u, x_noised)
            mse, mape = eval(a_gt, b_gt, a_mls, b_mls)

            result_mse[1][k][n]  = mse
            result_mape[1][k][n] = mape


            a_ss, b_ss = subspace_id(u, x, 10)
            mse, mape = eval(a_gt, b_gt, a_ss, b_ss)


            result_mse[2][k][n]  = mse
            result_mape[2][k][n] = mape


        print("eval for noise ", noise_level)

    eig_vals = numpy.array(eig_vals)

    fig, ax = plt.subplots(3)

    y_mean = result_mse[0].mean(axis=-1)
    y_std  = result_mse[0].std(axis=-1)
    ax[0].plot(noise_levels, y_mean, label="least squares", color="blue")
    ax[0].fill_between(noise_levels, y_mean - y_std, y_mean + y_std, alpha=0.3, color='blue')

    y_mean = result_mse[1].mean(axis=-1)
    y_std  = result_mse[1].std(axis=-1)
    ax[0].plot(noise_levels, y_mean, label="qr identification", color="red")
    ax[0].legend()


    y_mean = result_mse[2].mean(axis=-1)
    y_std  = result_mse[2].std(axis=-1)
    ax[0].plot(noise_levels, y_mean, label="subspace identification", color="purple")
    ax[0].legend()

    y_mean = result_mape[0].mean(axis=-1)
    y_std  = result_mape[0].std(axis=-1)
    ax[1].plot(noise_levels, y_mean, label="least squares", color="blue")
    ax[1].fill_between(noise_levels, y_mean - y_std, y_mean + y_std, alpha=0.3, color='blue')
    ax[1].legend()


    eig_re   = eig_vals.real
    eig_imag = eig_vals.imag
    ax[2].scatter(eig_re, eig_imag, color="blue", s =1 )
    
    
    plt.show()