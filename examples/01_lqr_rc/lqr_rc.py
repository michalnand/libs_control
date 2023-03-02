import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mat_a = numpy.zeros((1, 1))
    mat_b = numpy.zeros((1, 1))

    tau = 1.5

    mat_a[0][0] = -1.0/tau
    mat_b[0][0] = 1.0/tau

    dt = 0.001 

    steps = 10000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [1.0] ] )
    r = numpy.array( [ [0.01] ]) 


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, dt=dt)


    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1.0, steps=10000)
    LibsControl.plot_open_loop_response(t_result, y_result, "results/open_loop_response", labels=["voltage [V]"])


    
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
    xr = numpy.array([[1.0]]).T

    #step response
    u_result, x_result, y_result, = lqr.closed_loop_response(xr, steps)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, "results/closed_loop_response.png", ["voltage [V]"], ["voltage [V]"] )

