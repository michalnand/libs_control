from .sparse_solver     import *
from .non_linear_lib    import *

def polynom_fit(y, x, max_order):

    solver = SparseSolver()

    x_aug = polynome_augmentation(x, max_order)


    theta = numpy.linalg.lstsq(x_aug, y, rcond=None)[0]

    return theta

    #theta, thetas, loss = solver.solve(y, x_aug, steps_count = max_order)

    #return theta, thetas, loss


def polynom_predict(x, polynome_model):
    order = polynome_model.shape[0]
    
    x_aug = polynome_augmentation(x, order - 1)

    y = x_aug@polynome_model    

    return y
