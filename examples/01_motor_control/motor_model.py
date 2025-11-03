import LibsControl
import numpy

class MotorModel(LibsControl.DynamicalSystem):

    def __init__(self, b, Kt, Ke, R, J, dt):
        
        a_tmp = -(b + Kt*Ke/R)/J
        b_tmp = Kt/(R*J)
        
        a_mat = numpy.array([[a_tmp]])
        b_mat = numpy.array([[b_tmp]])

        LibsControl.DynamicalSystem.__init__(self, a_mat, b_mat, None, dt)

        

