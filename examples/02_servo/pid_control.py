import numpy

import LibsControl
import servo_model


if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = servo_model.ServoModel(tau, k, dt)


    # print matrices
    print(str(ds))

    #kp = 0.1
    #ki = 0.01
    #kd = 0.0

    kp = 4.0
    ki = 0.0
    kd = 50.0 
    controller = LibsControl.PID(kp, ki, kd)

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []   

    
    x = numpy.zeros((ds.a.shape[0], 1))
    u = 0.0

    angle_req = 1000


    # main simulation
    for n in range(n_steps):
        if n > n_steps*0.7:
            xr = numpy.zeros(x.shape)
        elif n > n_steps*0.1:
            xr = numpy.ones(x.shape)
        else:
            xr = numpy.zeros(x.shape)

        xr[1, 0] = 0
        xr = xr*angle_req/(2.0*numpy.pi*360)       

        # PID takes scalar inputs
        u = controller.forward(xr[0, 0], x[0, 0], u)


        # process simulation step
        u_in = numpy.array([[u]])
        x, _ = ds.forward(x, u_in)

        
        # convert to degrees and RPM
        xr_res = xr*2.0*numpy.pi
        xr_res[0, 0]*= 360.0
        xr_res[1, 0]*= 60.0

        x_res = x*2.0*numpy.pi
        x_res[0, 0]*= 360.0
        x_res[1, 0]*= 60.0


        # log results
        t_result.append(n*dt)
        u_result.append(u_in[:, 0])
        xr_result.append(xr_res)
        x_result.append(x_res)


    LibsControl.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/pid_result.png", ["input"], ["angle [degrees]", "rpm"])
    
    u_cost = LibsControl.compute_cost(numpy.array(u_result))
    x_cost = LibsControl.compute_cost(numpy.array(xr_result) - numpy.array(x))
    
    print(round(u_cost, 3), round(x_cost, 3))

