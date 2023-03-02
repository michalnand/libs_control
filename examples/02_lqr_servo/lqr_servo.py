import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mat_a = numpy.zeros((2, 2))
    mat_b = numpy.zeros((2, 1))

    tau = 1.9
    amp = 3.1

    mat_a[0][0] = -1.0/tau
    mat_a[0][1] = 0.0

    mat_a[1][0] = 1.0
    mat_a[1][1] = 0.0

    mat_b[0][0] = amp/tau

    dt = 0.001

    steps = 10000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [0.0, 0.0], [0.0, 1.0] ] )
    r = numpy.array( [ [0.5] ]) 


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, dt=dt)


    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1.0, steps=10000)
    LibsControl.plot_open_loop_response(t_result, y_result, "results/open_loop_response", labels=["velocity rad/s", "position rad"])


    
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
    xr = numpy.array([[0.0, 1.0]]).T

    #step response
    u_result, x_result, y_result = lqr.closed_loop_response(xr, steps)

    LibsControl.plot_closed_loop_response(t_result, u_result, y_result, "results/closed_loop_response.png", ["voltage [V]"], ["velocity, [rad/s]", "position [rad]"] )
