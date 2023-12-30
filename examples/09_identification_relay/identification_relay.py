import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":


    tau = 230
    b   = 100

   
    
    mat_a = numpy.zeros((2, 2))
    mat_b = numpy.zeros((2, 1))
    mat_c = numpy.zeros((2, 2))

   
    mat_a[0][0] =  -tau/b
    mat_a[1][0] =  1.0

    mat_b[0][0] = b
   

    mat_c[0][0] = 1.0
    mat_c[1][1] = 1.0
   


    dt          = 1.0/1000.0
    
    steps       = 6000



    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt=dt)



   
    u_batch = []
    x_batch = []

    x    = numpy.zeros((mat_a.shape[0], 1))

    x_idx = [1]
    u_idx = [0] 
    x_amp = [1.0] 
    u_amp = [1.0]

    relay_control = LibsControl.Relay(x_idx, u_idx, x_amp, u_amp)


    for i in range(steps):  
        

        u = relay_control.step(x)
        
        '''
        if i%200 == 0:
            u  = numpy.random.randn(mat_b.shape[1], 1)
        '''

        u_batch.append(u[:, 0])
        x_batch.append(x[:, 0])

        x, y = ds.forward(x, u)
 
    
    u_batch = numpy.array(u_batch)
    x_batch = numpy.array(x_batch)

   
    #x_batch+= 0.0001*numpy.random.randn(x_batch.shape[0], x_batch.shape[1])


    print("shape = ", u_batch.shape, x_batch.shape)

    models, loss = LibsControl.identification(u_batch, x_batch, dt, 5)


    model = models[2]

    ab      = model.T
    order   = x_batch.shape[1]
    a_hat = ab[:, 0:order]
    b_hat = ab[:, order:]
    
    print("ground truth")
    print(numpy.round(ds.mat_a, 3))
    print(numpy.round(ds.mat_b, 3))
    print("\n\n")

    print("model")
    print(numpy.round(a_hat, 3))
    print(numpy.round(b_hat, 3))

    print(loss)


    plt.clf()
    plt.ylabel("position")
    plt.xlabel("time [s]")
    plt.plot(u_batch[:, 0], color="red", label="control")
    plt.plot(x_batch[:, 1], color="deepskyblue", label="position")
    plt.legend()
    plt.grid()
    plt.show()
    