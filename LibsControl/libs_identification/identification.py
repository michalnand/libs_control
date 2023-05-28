import numpy
from .sparse_solver  import *
from .non_linear_lib import *


def identification(u, x, dt, steps_count=20):

    #shift values
    u_now    = u[0:-1, :]    
    x_now    = x[0:-1, :]
    x_next   = x[1:, :]

    dx  = (x_next - x_now)/dt

    #concatenate into single matrix
    w   = numpy.hstack([x_now, u_now])


    solver             = SparseSolver()
    _, thetas, loss    = solver.solve(dx, w, steps_count)

    return thetas, loss

    '''
    ab = thetas[8]

    print(">>> THETAS = ", thetas.shape)

    for i in range(thetas.shape[0]):
        density  = 100*(numpy.abs(thetas[i]) > 10**-5).sum()/(thetas.shape[1]*thetas.shape[2])
        density  = numpy.round(density, 1)
        loss_    = numpy.round(loss[i], 1)
        plt.cla()
        plt.clf()
        plt.title("density = " + str(density) + "% " + "   loss = " + str(loss_))
        plt.imshow(thetas[i].T , cmap = 'magma' )
        plt.colorbar()
        plt.savefig("images/model_ " + str(i) + ".png")


    print("model ID\t\t\tdensity[%]\t\t\tloss")
    for i in range(thetas.shape[0]):
        density = (numpy.abs(thetas[i]) > 10**-5).sum()/(thetas.shape[1]*thetas.shape[2])
        print(i, "\t\t\t\t", round(density*100, 1), "\t\t\t\t", round(loss[i], 2))
        

    #split to A and B matrices
    ab = ab.T
    order   = dx.shape[1]
    a = ab[:, 0:order]
    b = ab[:, order:]

    return a, b
    '''
