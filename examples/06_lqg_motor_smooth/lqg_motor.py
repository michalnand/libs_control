import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    
    u_max           = 6.0                       #V, supply voltage
    i_stall         = 1.6                       #A, stall current
    torque_stall    = 0.57*0.0980665            #kg.cm to Nm, stall torque
    w_max           = 1000.0*2.0*numpy.pi/60.0  #rpm to rad/s, max speed
    j_wheel         = 0.00005                   #[kg*m^2], rotor+wheel inertia
    
    r   = u_max/i_stall
    k   = torque_stall*r/u_max
    mu  = (k/(r*w_max))*(u_max - k*w_max)

    
    mat_a = numpy.zeros((2, 2))
    mat_b = numpy.zeros((2, 1))
    mat_c = numpy.zeros((2, 2))

    mat_a[0][0] = 0.0
    mat_a[1][1] = (1.0/j_wheel)*((k/r)*(-k) -mu)
    mat_a[1][0] = (1.0/j_wheel)*(k/r)
    
    mat_b[0][0] = 1.0

    mat_c[0][0] = 1.0
    mat_c[1][1] = 1.0

    observation_noise = 0.1

 
    dt = 1.0/256.0

    steps = 100

    t_result = numpy.arange(steps)*dt

    q = numpy.array([  [u_max**2, 0.0], [0.0, 1.0] ] )
    r = numpy.array( [ [0.1] ]) 

    w = (observation_noise**2)*numpy.eye(mat_c.shape[0])


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt=dt)

    print(str(ds))


    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = u_max, steps=steps)

    y_result[:, 1]*= 60.0/(2.0*numpy.pi)
    LibsControl.plot_open_loop_response(t_result, y_result, "results/open_loop_response", labels=["voltage [V]", "rpm"])
    

    
    lqg     = LibsControl.LQGSolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, w, dt)

    k, g, f = lqg.solve()

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", k, "\n")
    print("g=\n", g, "\n")
    print("f=\n", f, "\n")
    print("\n\n")


    
    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = lqg.get_poles()
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")

    
    #required state
    xr = numpy.array([[0.0, w_max]]).T

    #step response
    u_result, x_result, x_hat_result, y_result  = lqg.closed_loop_response(xr, steps, disturbance=True)
    x_hat_result[:, 1]*= 60.0/(2.0*numpy.pi)
    y_result[:, 1]*= 60.0/(2.0*numpy.pi)
    LibsControl.plot_closed_loop_response(t_result, u_result, y_result, x_hat_result, file_name = "results/closed_loop_response.png", u_labels=["control"], x_labels=["voltage [V]", "rpm"] )
    