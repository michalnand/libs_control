import numpy

class AnalyticalMPCDirect:
    def __init__(self, A, B, Q, R, prediction_horizon=16, control_horizon=4, u_max=1e10):
        """
        A: (n_x, n_x)
        B: (n_x, n_u)
        Q: (n_x, n_x) (state cost)
        R: (n_u, n_u) (input cost)
        prediction_horizon: N (how many future states)
        control_horizon: Nh (how many future inputs we optimize; typically <= N)
        """
        self.A = A
        self.B = B
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.N = prediction_horizon
        self.Nh = control_horizon
        self.u_max = u_max

        # build Phi and Theta (Theta maps the **sequence of future inputs U** to future stacked states)
        self.Phi, self.Theta = self._build_prediction_matrices(A, B, self.N, self.Nh)

        # build augmented Q and R   
        self.Q_aug = numpy.kron(numpy.eye(self.N), Q)  # block-diagonal
        self.R_aug = numpy.kron(numpy.eye(self.Nh), R)

        # Precompute solver matrices: H and Sigma = H^{-1} Theta^T Q_aug
        H = self.Theta.T @ self.Q_aug @ self.Theta + self.R_aug
        # use solve later for stability; but precompute factorization if desired
        # here we compute Sigma by solving H Sigma^T = Theta^T Q_aug  (do via solve)
        # Sigma has shape (n_u*Nh, n_x*N)
        # Solve H @ Sigma = Theta.T @ Q_aug  --> Sigma = numpy.linalg.solve(H, Theta.T @ Q_aug)
        self.Sigma = numpy.linalg.solve(H, self.Theta.T @ self.Q_aug)
        self.Sigma0 = self.Sigma[:self.nu, :]   # Only first-control block

    def _build_prediction_matrices(self, A, B, N, Nh):
        nx = A.shape[0]
        nu = B.shape[1]
        # Precompute A powers: A^0 ... A^N
        A_pows = [numpy.eye(nx)]
        for i in range(1, N + 1):
            A_pows.append(A_pows[-1] @ A)

        # Phi: (nx*N, nx) stacked [A; A^2; ...; A^N]
        Phi = numpy.zeros((nx * N, nx))
        for i in range(N):
            Phi[i * nx:(i + 1) * nx, :] = A_pows[i + 1]  # A^(i+1)

        # Theta: (nx*N, nu*Nh) where block (i,j) is A^(i-j) B for i>=j, else 0
        Theta = numpy.zeros((nx * N, nu * Nh))
        for i in range(N):
            for j in range(Nh):
                if i >= j:
                    # A^{i-j} B
                    Theta[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = A_pows[i - j] @ B
                else:
                    # remains zero
                    pass
        return Phi, Theta

    def forward_traj(self, Xr, x):
        """
        Computes only the first control action u0.
        """
        x = numpy.atleast_2d(x).reshape(self.nx, 1)
        Xr = numpy.atleast_2d(Xr).reshape(self.nx * self.N, 1)

        # residual
        s = Xr - self.Phi @ x

        # compute only first control
        u0 = self.Sigma0 @ s
        u0 = numpy.clip(u0, -self.u_max, self.u_max)

        print(self.Phi.shape, self.Sigma0.shape, s.shape, u0.shape)

        return u0