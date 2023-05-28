import numpy


class SparseSolver:

    '''
        x, input data,      shape = (n, x_features_count)
        y, target output, shape = (n, y_features_count)

        steps_count : num of solver iterations

        returns sparse model A, y = x@A

        matrix A is as sparse as possible
    '''
    def solve(self, y, x, steps_count = 20):
        y_tmp   = y.copy()

        theta   = numpy.zeros((x.shape[1], y.shape[1]), dtype=numpy.float32)
        thetas  = numpy.zeros((steps_count, x.shape[1], y.shape[1]), dtype=numpy.float32)
        
        loss    = numpy.zeros(steps_count)

        sparsity = 1.0 - 1.0/steps_count

        for i in range(steps_count):

            #solve by matrix pseudinverse and sparisfy model
            theta_          = self._solve(y_tmp, x)
            theta_sparse    = self._sparse(theta_, sparsity)

            #store solutions
            theta     = theta + theta_sparse
            thetas[i] = theta

            #prediction step
            y_predicted  = x@theta

            #residuum is input for next solution
            y_tmp        = y - y_predicted

            #store loss
            loss[i] = (y_tmp**2).mean()

             
        return theta, thetas, loss
        

    def _solve(self, y, x):
        #faster, numericaly more stable
        theta = numpy.linalg.lstsq(x, y, rcond=None)[0]

        '''
        #moore-penrose pseudoinverse
        x_inv   = numpy.linalg.pinv(x)
 
        #compute solution
        theta   = (x_inv@y)
        '''

        return theta
    
    def _sparse(self, x, sparsity):
        
        x_ = x.flatten()
        x_ = numpy.abs(x_)

        top_count = (1.0 - sparsity)*x_.shape[0]
        top_count = round(max(1.0, top_count))
        idx = numpy.argsort(x_)[::-1][:top_count][-1]

        mask = 1.0*(x_ >= x_[idx])
        mask = mask.reshape(x.shape)

        x_masked = x*mask 

        return x_masked

