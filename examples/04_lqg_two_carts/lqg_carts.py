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
    mat_c = numpy.zeros((2, 4))
        


    mat_a[0][2] =  1.0
    mat_a[1][3] =  1.0

    mat_a[2][0] =  -k/m1
    mat_a[2][1] =   k/m1
    mat_a[2][2] = -c1/m1

    mat_a[3][0] =   k/m2
    mat_a[3][1] =   -k/m2
    mat_a[3][3] = -c2/m2

    mat_b[2][0] = 1.0/m1


    mat_c[0][0] = 1.0
    mat_c[1][1] = 1.0
    


    dt          = 1.0/256
    
    noise = 0.01 

    steps = 2000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([ [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0] ] )
    r = numpy.array( [ [0.001] ]) 

    #measurement noise matrix
    w = (noise**2)*numpy.eye(mat_c.shape[0])


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt=dt)

    print(str(ds))

    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1, steps=steps)
    LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response", ["x0 [m]", "x1 [m]", "v0 [m/s]", "v1 [m/s]"])


 
    lqg     = LibsControl.LQGSolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, w, dt)

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
    xr = numpy.array([[0.0, 1.0, 0.0, 0.0]]).T

    #step response
    u_result, x_result, x_hat_result, y_result = lqg.closed_loop_response(xr, steps, noise = noise, disturbance = True)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat_result, "results/closed_loop_response.png", ["force [N]"], ["x0 [m]", "x1 [m]", "v0 [m/s]", "v1 [m/s]"] )
    LibsControl.plot_closed_loop_response(t_result, u_result, y_result, x_hat_result[:, 0:2], "results/closed_loop_response_observed.png", ["force [N]"], ["x0 [m]", "x1 [m]"] )
    