import LibsControl
import numpy
import matplotlib.pyplot as plt


from differential_robot import *

if __name__ == "__main__":

    dt    = 1.0/256.0
    model = DifferentialRobot(dt)

  
    steps = 5000

    t_result = numpy.arange(steps)*dt
    
    

    q = [0.0, 1.0, 0.0, 10.0]
    q = numpy.diag(q)

    
    r = [0.001, 0.001] 
    r =  numpy.diag(r)


    ds   = LibsControl.DynamicalSystem(model.mat_a, model.mat_b, model.mat_c, dt=dt)

    print(str(ds))

    
    #plot system open loop step response
    u_result, x_result, y_result = ds.step_response(amplitude = [1, -0.5], steps=steps)
    LibsControl.plot_open_loop_response(t_result, x_result, "results/open_loop_response",  labels = ["dtheta [deg/s]", "theta [deg]", "dx [m/s]", "x [m]"])


    
    lqri     = LibsControl.LQRISolver(ds.mat_a, ds.mat_b, ds.mat_c, q, r, dt)

    k, ki    = lqri.solve() 

    
    #print solved controller matrices
    print("controller\n\n")
    print("k=\n", numpy.round(k, 5), "\n")
    print("ki=\n", numpy.round(ki, 5), "\n")
    print("\n\n")

    '''
    #plot poles, both : open and closed loop
    re_ol, im_ol, re_cl, im_cl = LibsControl.get_poles(ds.mat_a, ds.mat_b, k)
    LibsControl.plot_poles(re_ol, im_ol, re_cl, im_cl, "results/poles.png")
    
    ranges, poles = LibsControl.get_poles_mesh(ds.mat_a, ds.mat_b, ds.mat_c)
    LibsControl.plot_poles_mesh(ranges, poles, "results/poles_mesh_ol.png")

    ranges, poles = LibsControl.get_poles_mesh(ds.mat_a - ds.mat_b@k, ds.mat_b, ds.mat_c)
    LibsControl.plot_poles_mesh(ranges, poles, "results/poles_mesh_cl.png")
    '''
    
    
    #required state 
    yr = numpy.array([[0.0, 90.0*numpy.pi/180.0, 0.0, 1.0]]).T

    #step response
    u_result, x_result, y_result = lqri.closed_loop_response(yr, steps, noise = 0, disturbance = True)

    x_result[:, 0]*= 180.0/numpy.pi
    x_result[:, 1]*= 180.0/numpy.pi
    
    LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name = "results/closed_loop_response.png", u_labels = ["uL [N]", "uR [N]"], x_labels = ["dtheta [deg/s]", "theta [deg]", "dx [m/s]", "x [m]"] )
    

    current_target     = 0
    
   

    target_x_pos = []
    target_y_pos = []

    targets = 30
    r       = 1.2 

    theta   = 0
    for i in range(targets):
        x = r*numpy.cos(theta)
        y = r*numpy.sin(theta)
        target_x_pos.append(x)
        target_y_pos.append(y)

        theta+= 1.618

    target_x_pos = numpy.array(target_x_pos)
    target_y_pos = numpy.array(target_y_pos)

    
    n = model.mat_a.shape[0]  #system order
    m = model.mat_b.shape[1]  #inputs count
    k = model.mat_c.shape[0]  #outputs count


    x       = numpy.zeros((n, 1))
    y       = numpy.zeros((k, 1))
    u       = numpy.zeros((m, 1))


    error_sum = numpy.zeros((k, 1))
    
    steps = 0
    while(True):

        
           

        x_pos    = target_x_pos[current_target]
        y_pos    = target_y_pos[current_target]
        distance = (x_pos - model.x_pos)**2 + (y_pos - model.y_pos)**2
        distance = distance**0.5

        required_angle    = numpy.arctan2(y_pos - model.y_pos, x_pos - model.x_pos)
        required_angle    = numpy.mod(required_angle, 2.0*numpy.pi)
        robot_angle       = model.theta

        d_angle = required_angle - robot_angle

        
        if d_angle < 0:
            d_angle+= 2.0*numpy.pi
        if d_angle > numpy.pi:
            d_angle-= 2.0*numpy.pi

        if distance < 0.05:
            current_target = (current_target + 1)%target_x_pos.shape[0]

        #if numpy.abs(d_angle) > 1*numpy.pi/180.0:
        #    distance = 0.0
        
        yr = numpy.array([[0.0, x[1, 0] + d_angle, 0.0,  x[3, 0] + distance]]).T

        u, error_sum = lqri.forward(yr, y, x, error_sum)
                 
        x, y = model.forward(x, u)

        if steps%10 == 0:
            model.render(x_pos, y_pos)
        
        
        steps+= 1
    

