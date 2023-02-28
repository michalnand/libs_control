import numpy
import matplotlib.pyplot as plt

'''
creates discrete dynamical system from 
continuous model

x(n+1) = a@x(n) + b@u(n)

a.shape = (system_order, system_order)
b.shape = (system_order, inputs_count)

x.shape = (system_order, 1)
u.shape = (inputs_count, 1)
'''

class DynamicalSystem:


    def __init__(self, mat_a, mat_b, dt):

        self.mat_a = mat_a
        self.mat_b = mat_b
        self.dt    = dt

    def __repr__(self) -> str:
        result = "continuous model : \n\n"
        result+= "a = \n"
        for j in range(self.mat_a.shape[0]):
            for i in range(self.mat_a.shape[1]):
                result+= str(round(self.mat_a[j][i], 5)) + " "
            result+= "\n"
        result+= "\n"

        result+= "b = \n"
        for j in range(self.mat_b.shape[0]):
            for i in range(self.mat_b.shape[1]):
                result+= str(round(self.mat_b[j][i], 5)) + " "
            result+= "\n"
        result+= "\n"

        return result


    def step_response(self, amplitudes, steps = 1000):
        x        = numpy.zeros((self.mat_a.shape[0], 1))
        u        = amplitudes*numpy.ones((self.mat_b.shape[1], 1))

        x_result = numpy.zeros((steps, self.mat_a.shape[0]))
        u_result = numpy.zeros((steps, self.mat_b.shape[0]))

  
        for n in range(steps):
            #system dynamics step

            x     = x + (self.mat_a@x + self.mat_b@u)*self.dt

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]

        return u_result, x_result
    
