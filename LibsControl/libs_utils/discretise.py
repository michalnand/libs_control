import numpy

def discretise(mat_a, mat_b, mat_c, dt):

    i         = numpy.eye(mat_a.shape[0])
    mat_a_inv = (i - 0.5*dt*mat_a)
    mat_a_inv = numpy.pinv(mat_a_inv)

    mat_ad    = mat_a_inv@(i - 0.5*dt*mat_a)
    mat_bd    = (mat_a_inv@mat_b)*dt
    mat_cd    = mat_c


    return mat_ad, mat_bd, mat_cd