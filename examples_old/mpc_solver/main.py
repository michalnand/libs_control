import numpy

def fast_gradient_method(A, y, x_min, x_max, 
                         x0=None, 
                         max_iter=100, 
                         L=None, 
                         beta=0.95, 
                         verbose=False):
    """
    Fast Gradient Method (Nesterov) for:
        minimize ||A x - y||^2
        s.t. x_min <= x <= x_max

    Parameters
    ----------
    A : (m, n) ndarray
    y : (m,) ndarray
    x_min, x_max : (n,) ndarray or scalars
    x0 : (n,) ndarray, optional initial guess
    max_iter : int, number of iterations
    L : float, Lipschitz constant of grad L(x) (optional)
    beta : float, momentum (optional, overrides auto)
    verbose : bool, print convergence info

    Returns
    -------
    x : ndarray, final solution
    history : list of cost values (optional)
    """

    m, n = A.shape
    if x0 is None:
        x = numpy.zeros(n)
    else:
        x = x0.copy()
    x_prev = x.copy()

    # Precompute reusable terms
    Q = A.T @ A
    b = A.T @ y

    # Estimate Lipschitz constant if not provided
    if L is None:
        L = 2 * numpy.linalg.norm(A, 2)**2   # 2 * (sigma_max(A))^2

    
    eta = 1.0 / L


    history = []
    for k in range(max_iter):
        # Momentum coefficient

        '''
        if beta is None:
            beta = (k - 1) / (k + 2) if k > 0 else 0.0
        else:
            beta = beta
        '''

        if k > 0:
            beta_k = 0.0
        else:
            beta_k = beta

        # Extrapolation
        y_k = x + beta_k * (x - x_prev)

        # Gradient at extrapolated point
        grad = 2 * (Q @ y_k - b)

        # Gradient step + projection (clamp)
        x_next = numpy.clip(y_k - eta * grad, x_min, x_max)

        if verbose:
            cost = numpy.sum((A @ x_next - y)**2)
            history.append(cost)
            print(f"Iter {k:3d}, cost={cost:.4e}")

        # Update
        x_prev, x = x, x_next

    return x, history


if __name__ == "__main__":

    dim_x = 17
    dim_y = 5

    y = numpy.random.randn(dim_y)
    A = numpy.random.randn(dim_y, dim_x)

    x_max = 1000*numpy.ones(dim_x)
    x_min = -1000*numpy.ones(dim_x)

    x, h = fast_gradient_method(A, y, x_min, x_max, verbose=True)

    print(x)
    

