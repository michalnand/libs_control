import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":


    dt = 0.01

    #time constant for yaw angle robot rotation (steering)
    tau_turn    = 6.3

    #time constant for robot acceleration (forward direction)
    tau_forward = 0.9

    #amplification, ratio between measured value amplitude : controll variable amplitude
    b_turn      = 1.5
    b_forward   = 0.2



    mat_a = numpy.zeros((4, 4))
    mat_b = numpy.zeros((4, 2))
    mat_c = numpy.eye(4)

    mat_a[0][0] =  -tau_turn
    mat_a[1][0] =  1.0
    mat_a[2][2] = -tau_forward
    mat_a[3][2] = 1.0


    mat_b[0][0] =  -b_turn*tau_turn
    mat_b[0][1] =   b_turn*tau_turn

    mat_b[2][0] =  b_forward*tau_forward
    mat_b[2][1] =  b_forward*tau_forward


    disc_mat_a, disc_mat_b, disc_mat_c = LibsControl.discretise(mat_a, mat_b, mat_c, dt)


    print("continuous model \n")
    print(mat_a)
    print()
    print(mat_b)
    print("\n")

    print("discrete model \n")
    print(disc_mat_a)
    print()
    print(disc_mat_b)
    print("\n")


      
    x_log = []
    u_log = []

    n_states = disc_mat_a.shape[0]
    n_inputs = disc_mat_b.shape[1]

    x = numpy.zeros((disc_mat_a.shape[0], 1))
    
    u = numpy.zeros((n_inputs, 1))
    for n in range(10000):
        
        if numpy.random.rand() < 0.1:
            u = numpy.random.randint(0, 3, (n_inputs, 1))-1

        x_next = disc_mat_a@x + disc_mat_b@u

        x_log.append(x[:, 0])
        u_log.append(u[:, 0])

        x = x_next.copy()



    x_log = numpy.array(x_log)
    u_log = numpy.array(u_log)

    noise_level = 0.01
    x_log+= noise_level*numpy.random.randn(x_log.shape[0], x_log.shape[1])

    mat_a_hat, mat_b_hat = LibsControl.recurisve_ls_identification(u_log, x_log) 

    print("predicted model \n")
    print(numpy.round(mat_a_hat, 4))
    print()
    print(numpy.round(mat_b_hat, 4))
    print("\n")  
    print("error = ", ((disc_mat_a - mat_a_hat)**2).mean() + ((disc_mat_b - mat_b_hat)**2).mean() )
    print("\n\n")

    R = numpy.eye(mat_a.shape[0])*noise_level
    Q = numpy.zeros((mat_a.shape[0] + mat_b.shape[1], mat_a.shape[0] + mat_b.shape[1]))

    mat_a_hat, mat_b_hat = LibsControl.recurisve_kalman_ls_identification(u_log, x_log, R, Q)    

    print("predicted model \n")
    print(numpy.round(mat_a_hat, 4))
    print()
    print(numpy.round(mat_b_hat, 4))
    print("\n")
    print("error = ", ((disc_mat_a - mat_a_hat)**2).mean() + ((disc_mat_b - mat_b_hat)**2).mean() )
    print("\n\n")