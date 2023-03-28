import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    c0 = 2.697*1e-3
    c1 = 2.66 *1e-3
    c2 = 3.05 *1e-3
    c3 = 2.85 *1e-3

    d0 = 6.78*1e-5
    d1 = 8.01*1e-5
    d2 = 8.82*1e-5

    J  = 2.25e-4
    
    mat_a = numpy.zeros((6, 6))
    mat_b = numpy.zeros((6, 2))
    mat_c = numpy.zeros((6, 6))


        
    mat_a[0][3] =  1.0
    mat_a[1][4] =  1.0
    mat_a[2][5] =  1.0

    mat_a[3][0] =  (-c0-c1)/J
    mat_a[3][1] =   c1/J
    mat_a[3][3] =  -d0/J

    mat_a[4][0] =  c1/J
    mat_a[4][1] =  (-c1-c2)/J
    mat_a[4][2] =  c2/J
    mat_a[4][4] =  -d1/J

    mat_a[5][1] =   c2/J
    mat_a[5][2] =   -c2/J
    mat_a[5][3] =   -c3/J
    mat_a[5][5] =  -d2/J

   
    mat_b[3][0] = c0/J
    mat_b[5][1] = c3/J
   

    mat_c[0][0] = 1.0
    mat_c[1][1] = 1.0
    mat_c[2][2] = 1.0
    mat_c[3][3] = 1.0
    mat_c[4][4] = 1.0
    mat_c[5][5] = 1.0
    


    dt          = 1.0/1000.0
    
    noise = 0.01 

    steps = 8000

    t_result = numpy.arange(steps)*dt

    q = numpy.array([   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ] )
    
    r = numpy.array( [ [0.001, 0.000], [0.000, 0.001] ]) 


    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt=dt)

    print(str(ds))

    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = 1, steps=steps)
    LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response", ["disc A [rad]", "disc B [rad]", "disc C [rad]", "disc A [rad/s]", "disc B [rad/s]", "disc C [rad/s]"])


 
    lqri     = LibsControl.LQRISolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, dt)

    k, ki    = lqri.solve()

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", k, "\n")
    print("ki=\n", ki, "\n")
    print("\n\n")

    
    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = lqri.get_poles()
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")

    
    #required state
    xr = numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

    #step response
    u_result, x_result, y_result = lqri.closed_loop_response(xr, steps, noise = noise, disturbance = True)

    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name = "results/closed_loop_response.png", u_labels = ["motor A", "motor B"], x_labels = ["disc A [rad]", "disc B [rad]", "disc C [rad]", "disc A [rad/s]", "disc B [rad/s]", "disc C [rad/s]"] )
    