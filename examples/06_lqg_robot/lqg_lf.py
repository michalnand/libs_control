import LibsControl
import numpy
import matplotlib.pyplot as plt


from robot_model import *
from robot_test  import *

if __name__ == "__main__":

    dt      = 1.0/256.0
    steps   = int(10.0*(1.0/dt))
    observation_noise = 0.1 
    

    t_result = numpy.arange(steps)*dt

    ds  = RobotModel(dt=dt)

    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitudes = [[1.1], [0.9]], steps=steps)

    x_result[:, 1]*= (180.0/numpy.pi)
    x_result[:, 3]*= (180.0/numpy.pi)
    #LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response", ["velocity m/s", "angular rate [deg/s]", "position [m]", "angle [deg]"])

  

    q = numpy.array([ [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0] ] )
    r = numpy.array( [ [1.0, 0.0], [0.0, 1.0] ]) 

    #measurement noise matrix
    v = (observation_noise**2)*numpy.eye(ds.mat_c.shape[0])

    #model uncertaininty matrix
    w = (0.01)*numpy.eye(ds.mat_a.shape[0])

    
    lqg     = LibsControl.LQGSolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, v, w, dt)

    k, g, f = lqg.solve()

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", k, "\n")
    print("g=\n", g, "\n")
    print("f=\n", f, "\n")
    print("\n\n")

 
    
    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = lqg.get_poles()
    #LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")

     
    #required state
    xr = numpy.array([[0.0, 0.0, 4.0, 90.0*numpy.pi/180.0]]).T

    #step response
    u_result, x_result, x_hat_result, y_result = lqg.closed_loop_response(xr, steps, observation_noise = observation_noise, disturbance = True)

    x_hat_result[:, 1]*= (180.0/numpy.pi)
    x_hat_result[:, 3]*= (180.0/numpy.pi)
     
    x_result[:, 1]*= (180.0/numpy.pi)
    x_result[:, 3]*= (180.0/numpy.pi) 

    y_result[:, 1]*= (180.0/numpy.pi)
    y_result[:, 3]*= (180.0/numpy.pi)

    #LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat_result, "results/closed_loop_response.png", ["u right", "u left"], ["velocity m/s", "angular rate [deg/s]", "position [m]", "angle [deg]"] )
    #LibsControl.plot_closed_loop_response(t_result, u_result, y_result, x_hat_result, "results/closed_loop_response_observed.png", ["u right", "u left"], ["velocity m/s", "angular rate [deg/s]", "position [m]", "angle [deg]"] )


    while True:
        u = 0*numpy.random.randn(2, 1)
        ds.step(u)
        ds.render()