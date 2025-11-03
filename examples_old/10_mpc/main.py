import numpy
import LibsControl
import matplotlib.pyplot as plt



# create required trajectory - square motion pattern
def get_required_x(path_length = 500):

    xr_trajectory = []


    for j in range(1):
        for i in range(path_length):
            x = [2.0*i/path_length - 1.0, 0, -1.0, 0]
            xr_trajectory.append(x)

        for i in range(path_length):
            x = [1.0, 0, 2.0*i/path_length - 1.0, 0]
            xr_trajectory.append(x)

        for i in range(path_length):
            x = [-(2.0*i/path_length - 1.0), 0, 1.0, 0]
            xr_trajectory.append(x)

        for i in range(path_length):
            x = [-1.0, 0, -(2.0*i/path_length - 1.0), 0]
            xr_trajectory.append(x)


    xr_trajectory = numpy.array(xr_trajectory, dtype=numpy.float32)

    return xr_trajectory


def mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, Optimizer): 
    if Optimizer is None:
        mpc = LibsControl.AnalyticalMPC(a_disc, b_disc, q, r, prediction_horizon, control_horizon)
    else:
        mpc = LibsControl.GradientMPC(a_disc, b_disc, q, r, prediction_horizon, control_horizon, Optimizer)

    print("solving with optimizer ", Optimizer)
    
    # obtain required trajectory
    xr_trajectory = get_required_x()
    n_max = xr_trajectory.shape[0]


    #result log
    mpc_t_result = []
    mpc_u_result = []
    mpc_x_result = []


    #initial motor state
    ds.reset()

    ds.x[0, 0] = xr_trajectory[0, 0]
    ds.x[2, 0] = xr_trajectory[0, 2]

    u = numpy.zeros((mat_b.shape[1], 1))
    
    for n in range(n_max-prediction_horizon):
        #plant state
        x = ds.x

        #compute controller output
        xr = xr_trajectory[n:n+prediction_horizon, :]
        u = mpc.forward(xr, x, u)

        #compute plant output   
        x, y = ds.forward_state(u)

        # store trajectory
        mpc_t_result.append(n*dt)
        mpc_u_result.append(u[:, 0].copy())
        mpc_x_result.append(x[:, 0].copy())
        
        
    mpc_t_result = numpy.array(mpc_t_result)
    mpc_x_result = numpy.array(mpc_x_result)
    mpc_u_result = numpy.array(mpc_u_result)

    return mpc_t_result, mpc_x_result, mpc_u_result




if __name__ == "__main__":


    dt = 1.0/100

    mat_a = numpy.zeros((4, 4))
    mat_b = numpy.zeros((4, 2))

    tau_a = 0.05
    tau_b = 0.5

    mat_a[0][1] = 1.0
    mat_a[1][1] = -1.0/tau_a
    mat_a[2][3] = 1.0
    mat_a[3][3] = -1.0/tau_b

    mat_b[1][0] = 1.0/tau_a
    mat_b[3][1] = 1.0/tau_b


    mat_c = numpy.eye(mat_a.shape[0])


    #mat_a = numpy.random.randn(4, 4)
    #mat_b = numpy.random.randn(4, 2)


    #create dynamical system
    ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

    tau_min, tau_max = ds.find_min_max_tau()


    a_disc, b_disc, c_disc = LibsControl.c2d(mat_a, mat_b, mat_c, dt)

    print(mat_a)
    print(a_disc)
    print(b_disc)

    prediction_horizon = int(tau_max/dt)
    control_horizon    = int(tau_min/dt)

    print("tau = ", tau_min, tau_max)
    print("prediction_horizon = ", prediction_horizon)
    print("control_horizon    = ", control_horizon)


    #create loss weighting matrices (diagonal)
    q = numpy.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] )
    r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 

    
    xr_trajectory = get_required_x()
    _, a_x_result, _ = mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, None)

    _, sgd_x_result, _ = mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, LibsControl.OptimSGD)
    _, fgm_x_result, _ = mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, LibsControl.OptimFGM)

    _, adam_x_result, _ = mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, LibsControl.OptimAdam)
    _, adopt_x_result, _ = mpc_test(ds, a_disc, b_disc, q, r, prediction_horizon, control_horizon, LibsControl.OptimAdopt)


  

    plt.plot(xr_trajectory[:, 0], xr_trajectory[:, 2], label="required trajectory", color= "red", lw=3)
    plt.plot(a_x_result[:, 0], sgd_x_result[:, 2], label="MPC analytical", color= "black", lw=3, alpha=0.8)

    plt.plot(sgd_x_result[:, 0], sgd_x_result[:, 2], label="MPC SGD", color= "springgreen", lw=2, alpha=0.8)
    plt.plot(fgm_x_result[:, 0], fgm_x_result[:, 2], label="MPC FGM", color= "orange", lw=2, alpha=0.8)
    plt.plot(adam_x_result[:, 0], adam_x_result[:, 2], label="MPC adam", color= "blueviolet", lw=2, alpha=0.8)
    plt.plot(adopt_x_result[:, 0], adopt_x_result[:, 2], label="MPC adopt", color= "blue", lw=2, alpha=0.8)

    plt.legend()
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    plt.show()
