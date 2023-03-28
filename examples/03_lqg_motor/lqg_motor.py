import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mat_a = numpy.zeros((3, 3))
    mat_b = numpy.zeros((3, 1))

    J = 0.02
    b = 0.2
    K = 0.3
    R = 2.0
    L = 0.4
    
    mat_a[0][1] = 1.0
    
    mat_a[1][1] = -b/J
    mat_a[1][2] = K/J
    
    mat_a[2][1] = -K/J
    mat_a[2][2] = -R/L

    mat_b[2][0] = 1.0/L

    dt = 0.001 

    steps = 10000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
    r = numpy.array( [ [0.001] ]) 
    
    noise = 0.25

    w = noise*numpy.eye(mat_a.shape[0])


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, dt=dt)

    print(str(ds))


    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1.0, steps=10000)
    LibsControl.plot_open_loop_response(t_result, y_result, "results/open_loop_response", labels=["angle", "angular velocity", "current"])

    
    lqg = LibsControl.LQGSolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, w, dt = dt)

    k, ki, f    = lqg.solve()

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", k, "\n")
    print("ki=\n", ki, "\n")
    print("f=\n", f, "\n")
    print("\n\n")


    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = lqg.get_poles()
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")


    #required state
    xr = numpy.array([[100.0*numpy.pi/180.0, 0.0, 0.0]]).T
 
    #step response
    u_result, x_result, x_hat_result, y_result = lqg.closed_loop_response(xr, steps, noise=noise, disturbance = True)

    x_result[:, 0]*= 180.0/numpy.pi
    x_result[:, 1]*= 60.0/(2.0*numpy.pi)

    x_hat_result[:, 0]*= 180.0/numpy.pi
    x_hat_result[:, 1]*= 60.0/(2.0*numpy.pi)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat=x_hat_result, file_name = "results/closed_loop_response.png", u_labels = ["voltage [V]"], x_labels = ["angle [degrees]", "angular velocity [rpm]", "current [A]"] )
    
    