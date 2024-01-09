import LibsControl
import numpy
import matplotlib.pyplot as plt


from balancing_robot import *

if __name__ == "__main__":

    dt    = 1.0/256.0
    model = BalancingRobot(dt)

  
    steps = 5000

    t_result = numpy.arange(steps)*dt
    
    

    q = [1.0, 0.0, 0.001, 0.0, 1.0, 0.0]
    q = numpy.diag(q)

    
    r = [0.1, 0.1] 
    r =  numpy.diag(r)


    ds   = LibsControl.DynamicalSystem(model.mat_a, model.mat_b, model.mat_c, dt=dt)

    print(str(ds))

    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = [1, 1], steps=steps)
    LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response",  labels = ["x [m]", "dx [m/s]", "theta [deg]", "dtheta [deg/s]", "phi [deg]", "dphi [deg]"])


    
    lqri     = LibsControl.LQRISolver(ds.mat_a, ds.mat_b, q, r, dt)

    k, ki    = lqri.solve() 

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", numpy.round(k, 5), "\n")
    print("ki=\n", numpy.round(ki, 5), "\n")
    print("\n\n")

    
    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = LibsControl.get_poles(ds.mat_a, ds.mat_b, k)
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")

    
    '''
    ranges, poles = LibsControl.get_poles_mesh(ds.mat_a, ds.mat_b, ds.mat_c)
    LibsControl.plot_poles_mesh(ranges, poles, "results/poles_mesh_ol.png")

    ranges, poles = LibsControl.get_poles_mesh(ds.mat_a - ds.mat_b@k, ds.mat_b, ds.mat_c)
    LibsControl.plot_poles_mesh(ranges, poles, "results/poles_mesh_cl.png")
    '''
    
    #required state 
    xr = numpy.array([[1.0, 0.0, 100.0*numpy.pi/180.0, 0.0, 0.0, 0.0]]).T

    #step response
    u_result, x_result = lqri.closed_loop_response(xr, steps, noise = 0, disturbance = True)

    x_result[:, 2]*= 180.0/numpy.pi
    x_result[:, 3]*= 180.0/numpy.pi
    x_result[:, 4]*= 180.0/numpy.pi
    x_result[:, 5]*= 180.0/numpy.pi
    
    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name = "results/closed_loop_response.png", u_labels = ["uL [N]", "uR [N]"], x_labels = ["x [m]", "dx [m/s]", "theta [deg]", "dtheta [deg/s]", "phi [deg]", "dphi [deg]"] )
    

    
    n = model.mat_a.shape[0]  #system order
    m = model.mat_b.shape[1]  #inputs count
    k = model.mat_a.shape[0]  #outputs count


    x       = numpy.zeros((n, 1))
    x_hat   = numpy.zeros((n, 1))
    y       = numpy.zeros((k, 1))
    u       = numpy.zeros((m, 1))

    error_sum = numpy.zeros((k, 1))
    
    steps = 0
    while(True):

        m = (steps//500)%10
        
        if m == 0:
            yr = numpy.array([[0.0, 0.0, 0.0]]).T
        elif m == 1:
            yr = numpy.array([[0.8, 0.0, 0.0]]).T
        elif m == 2:
            yr = numpy.array([[0.8, 0.0, 90.0*numpy.pi/180.0]]).T
        elif m == 3:
            yr = numpy.array([[0.8, 0.0, -90.0*numpy.pi/180.0]]).T
        elif m == 4:
            yr = numpy.array([[0.8, 0.0, 0.0]]).T

        elif m == 5:
            yr = numpy.array([[0.0, 0.0, 0.0]]).T
        elif m == 6:
            yr = numpy.array([[-0.8, 0.0, 0.0]]).T
        elif m == 7:
            yr = numpy.array([[-0.8, 0.0, -90.0*numpy.pi/180.0]]).T
        elif m == 8:
            yr = numpy.array([[-0.8, 0.0,  90.0*numpy.pi/180.0]]).T
        elif m == 9:
            yr = numpy.array([[-0.8, 0.0, 0.0]]).T

        xr = numpy.zeros((6, 1))

        xr[0, 0] = yr[0, 0]
        xr[2, 0] = yr[1, 0]
        xr[4, 0] = yr[2, 0]

        u, error_sum = lqri.forward(xr, x, error_sum)
                 
        x, y = model.forward(x, u)

        if steps%10 == 0:
            model.render(y)
           
        steps+= 1
    

