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
    
    steps       = 10000



    ds   = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt=dt)


    u_batch = []
    x_batch = []

    x = numpy.zeros((mat_a.shape[0], 1))


    u    = numpy.zeros((mat_b.shape[1], 1))

    for i in range(steps):  
        
        if i%200 == 0:
            u    = numpy.random.rand(mat_b.shape[1], 1)
        
        u_batch.append(u[:, 0])
        x_batch.append(x[:, 0])

        x, y = ds.forward(x, u)
 


    
    u_batch = numpy.array(u_batch)
    x_batch = numpy.array(x_batch)

    x_batch+= 0.00001*numpy.random.randn(x_batch.shape[0], x_batch.shape[1])


    a_hat, b_hat = LibsControl.identification(u_batch, x_batch, dt)

    
    #a_hat, b_hat = LibsControl.identification_cont(u_batch, x_batch, dt)

    print("ground truth")
    print(numpy.round(ds.mat_a, 3))
    print(numpy.round(ds.mat_b, 3))
    print("\n\n")

    print("model")
    print(numpy.round(a_hat, 3))
    print(numpy.round(b_hat, 3))
    