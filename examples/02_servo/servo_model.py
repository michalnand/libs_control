import LibsControl
import numpy

class ServoModel(LibsControl.DynamicalSystem):

    def __init__(self, tau, k, dt):
        
        a_tmp = -1.0/tau
        b_tmp = k*1.0/tau   
        
        a_mat = numpy.array([[0.0, 1.0], [0.0, a_tmp]])
        b_mat = numpy.array([[0.0], [b_tmp]])

        LibsControl.DynamicalSystem.__init__(self, a_mat, b_mat, None, dt)

        

