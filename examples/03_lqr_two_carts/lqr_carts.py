import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    m1 = 0.14 
    m2 = 0.17 
    c1 = 0.2 
    c2 = 0.3 
    k  = 1.9

    mat_a = numpy.zeros((4, 4))
    mat_b = numpy.zeros((4, 1))
        


    mat_a[0][2] =  1.0
    mat_a[1][3] =  1.0

    mat_a[2][0] =  -k/m1
    mat_a[2][1] =   k/m1
    mat_a[2][2] = -c1/m1

    mat_a[3][0] =   k/m2
    mat_a[3][1] =   -k/m2
    mat_a[3][3] = -c2/m2

    mat_b[2][0] = 1.0/m1


    f_sampling  = 32.0
    dt          = 1.0/(10.0*f_sampling)

    steps = 1200

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0] ] )
    r = numpy.array( [ [0.1] ]) 


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, dt=dt)


    #plot system open loop step response
    u_result, x_result = ds.step_response(amplitudes = 1, steps=steps)
    LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response", labels=["x0 [m]", "x1 [m]", "v0 [m/s]", "v1 [m/s]"])


    
    lqr = LibsControl.LQRSolver(ds.mat_a, ds.mat_b, q, r, dt)

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
    xr = numpy.array([[0.0, 1.0, 0.0, 0.0]]).T

    #step response
    u_result, x_result = lqr.closed_loop_response(xr, steps, 10)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, "results/closed_loop_response.png", ["force [N]"], ["x0 [m]", "x1 [m]", "v0 [m/s]", "v1 [m/s]"] )
    