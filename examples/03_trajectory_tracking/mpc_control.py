import numpy

import LibsControl
import position_model
import visualisation



def get_required(n, n_steps, prediction_horizon):
    Xr = []
    for h in range(prediction_horizon):

        turn_state = ((n + h)//(n_steps//4))%4

        if turn_state == 0:
            x_pos_req = 1.0
            y_pos_req = 0.0
        elif turn_state == 1:
            x_pos_req = 1.0
            y_pos_req = 1.0
        elif turn_state == 2:
            x_pos_req = 0.0
            y_pos_req = 1.0
        else:
            x_pos_req = 0.0
            y_pos_req = 0.0

        xr = numpy.zeros((ds.a.shape[0], 1))

        xr[0, 0] = x_pos_req
        xr[1, 0] = y_pos_req

        Xr.append(xr)

    Xr = numpy.vstack(Xr)

    return Xr


if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = position_model.PositionModel(tau, k, dt)

    # print matrices
    print(str(ds))  



    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = LibsControl.c2d(ds.a, ds.b, None, dt)

    print(a_disc)
    print(b_disc)

    q = numpy.diag([10000, 10000, 0.0, 0.0])
    r = numpy.diag([1.0, 1.0])  

    prediction_horizon = 64
    control_horizon    = 32
    controller = LibsControl.AnalyticalMPCDirect(a_disc, b_disc, q, r, prediction_horizon, control_horizon, 10)


    #print("k  = ", controller.k)
    #print("f  = ", controller.f)
    #print("phi   = ", controller.phi)
    #print("omega = ", controller.omega)
    #print("sigma = ", controller.sigma )



    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    x  = numpy.zeros((ds.a.shape[0], 1))
    u = numpy.zeros((ds.b.shape[1], 1))

    pos_req = 1.0

    x_pos_req = 0.0
    y_pos_req = 0.0

    # main simulation, steps for logs
    for n in range(n_steps):
        

        '''
        turn_state = (n//(n_steps//4))%4
        if turn_state == 0:
            x_pos_req = 1.0
            y_pos_req = 0.0
        elif turn_state == 1:
            x_pos_req = 1.0
            y_pos_req = 1.0
        elif turn_state == 2:
            x_pos_req = 0.0
            y_pos_req = 1.0
        else:
            x_pos_req = 0.0
            y_pos_req = 0.0


        xr = numpy.zeros((ds.a.shape[0], 1))

        xr[0, 0] = x_pos_req
        xr[1, 0] = y_pos_req
        '''

        Xr = get_required(n, n_steps, prediction_horizon)
        
        # PID takes scalar inputs   
        u = controller.forward_traj(Xr, x)

        # process simulation step
        x, _ = ds.forward(x, u)

      

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        xr_result.append(Xr[0:4, 0])
        x_result.append(x[:, 0])


    LibsControl.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/mpc_result.png",  ["input X", "input Y"],  ["position X", "position Y", "velocity X", "velocity Y"])
    

    ds.reset()


    visualisation = visualisation.TrajectoryRenderer()

    
    x_pos_req = 0.0
    y_pos_req = 0.0

    n = 0

    u[:] = 0

    
    # main simulation demo
    while True:

        Xr = get_required(n, n_steps, prediction_horizon)
        
        # PID takes scalar inputs   
        u = controller.forward_traj(Xr, x)

        # process simulation step
        x, _ = ds.forward(x, u) 

        n+= 1

        if n%2 == 0:    
            positions = []
            positions.append(Xr[0:2, 0])
            positions.append(x[0:2, 0])
            res = visualisation.step(positions, [[1.0, 1.0, 1.0], [0.8, 0.0, 0.2]], ["reference", "MPC"])

            if res != 0:
                break
    