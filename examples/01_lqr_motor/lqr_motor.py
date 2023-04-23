import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mat_a = numpy.zeros((3, 3))
    mat_b = numpy.zeros((3, 1))

    J = 0.02    #rotor moment of inertia (kg.m^2)
    b = 0.2     #drag coefficient (N/m/s)
    K = 0.3     #motor constant (N.m.A^-1)
    R = 2.0     #wiring resitance, (ohm)
    L = 0.4     #wiring inductance, (H)
    
    mat_a[0][1] = 1.0
    
    mat_a[1][1] = -b/J
    mat_a[1][2] = K/J
    
    mat_a[2][1] = -K/J
    mat_a[2][2] = -R/L

    mat_b[2][0] = 1.0/L

    dt = 0.001 

    steps = 10000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
    r = numpy.array( [ [0.001] ]) 


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, dt=dt)

    print(str(ds))


    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1.0, steps=10000)
    LibsControl.plot_open_loop_response(t_result, y_result, "results/open_loop_response", labels=["angle", "angular velocity", "current"])


    
    lqr = LibsControl.LQRSolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, dt)

    k, g    = lqr.solve()

    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", k, "\n")
    print("g=\n", g, "\n")
    print("\n\n")


    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = lqr.get_poles()
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")


    #required state
    xr = numpy.array([[100.0*numpy.pi/180.0, 0.0, 0.0]]).T

    #step response
    u_result, x_result, y_result, = lqr.closed_loop_response(xr, steps, disturbance = True)

    x_result[:, 0]*= 180.0/numpy.pi
    x_result[:, 1]*= 60.0/(2.0*numpy.pi)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name = "results/closed_loop_response.png", u_labels = ["voltage [V]"], x_labels = ["angle [degrees]", "angular velocity [rpm]", "current [A]"] )
    
    