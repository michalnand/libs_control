import numpy


'''
least squares identification for discrete system :

x(n+1) = x(n)A + u(n)B

input is transposed, batch-like : 

x.shape = (batch_size, N)
u.shape = (batch_size, M)

where N is system order, M is system inputs

returned matrices have shape : 
A.shape = (N, N)
B.shape = (N, M)
'''
def ls_identification(u, x):
    x_now  = x[0:-1, :]
    u_now  = u[0:-1, :]

    x_next = x[1:, :]

    x_tmp = numpy.hstack([x_now, u_now])

    theta = numpy.linalg.lstsq(x_tmp, x_next, rcond=None)[0]

    a = theta[0:x_now.shape[1], :]
    b = theta[x_now.shape[1]: , :]

    a = numpy.array(a.T)
    b = numpy.array(b.T)

    return a, b



'''
robust least squares identification for discrete system :

x(n+1) = x(n)A + u(n)B

input is transposed, batch-like : 

x.shape = (batch_size, N)
u.shape = (batch_size, M)

where N is system order, M is system inputs

returned matrices have shape : 
A.shape = (N, N)
B.shape = (N, M)

starts with least squares fitting, then keeps p_best fraction of best fits (1-p_best) throws away, 
and fits again this repeats n_iterations steps which effectively removes outliers
'''
'''
def rls_identification(u, x, n_iterations = 20, p_best = 0.99):

    x_now  = x[0:-1, :]
    u_now  = u[0:-1, :]
    x_next = x[1:, :]

    xu_sub      = numpy.hstack([x_now, u_now])

    x_next_sub  = x_next.copy()

    for n in range(n_iterations):

        theta = numpy.linalg.lstsq(xu_sub, x_next_sub, rcond=None)[0]
        error = ((x_next_sub - xu_sub@theta)**2).sum(axis=-1)

        #sort from lowest to biggest error
        indices = numpy.argsort(error)

        #select p_best ratio for next round (e.g. 90% best)
        cut_off_index = int(p_best*indices.shape[0])
        indices_best  = indices[0:cut_off_index]

        xu_sub = xu_sub[indices_best, :]
        x_next_sub = x_next_sub[indices_best, :]
        

    a = theta[0:x_now.shape[1], :]
    b = theta[x_now.shape[1]:,  :]

    a = numpy.array(a.T)
    b = numpy.array(b.T)
   
    return a, b
'''