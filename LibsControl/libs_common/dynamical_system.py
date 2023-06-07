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


    def __init__(self, mat_a, mat_b, mat_c = None, dt = 0.001):

        self.mat_a = mat_a
        self.mat_b = mat_b

        #choose full state output if not output matrix provided
        if mat_c is not None:
            self.mat_c = mat_c
        else:
            self.mat_c = numpy.eye(self.mat_a.shape[0])

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

        result+= "c = \n"
        for j in range(self.mat_c.shape[0]):
            for i in range(self.mat_c.shape[1]):
                result+= str(round(self.mat_c[j][i], 5)) + " "
            result+= "\n"
        result+= "\n" 

        return result

    def forward(self, x, u):        
        x_new   = x + (self.mat_a@x + self.mat_b@u)*self.dt
        y       = self.mat_c@x_new

        return x_new, y

    def step_response(self, amplitude, steps = 1000):
        x        = numpy.zeros((self.mat_a.shape[1], 1))

        amplitudes = numpy.array(amplitude)
        amplitudes = numpy.expand_dims(amplitudes, 1)
        u        = amplitudes*numpy.ones((self.mat_b.shape[1], 1))

        print(u)
        

        u_result = numpy.zeros((steps, self.mat_b.shape[1]))
        x_result = numpy.zeros((steps, self.mat_a.shape[0]))
        y_result = numpy.zeros((steps, self.mat_c.shape[0]))

  
        for n in range(steps):
            #system dynamics step

            x, y = self.forward(x, u)

            u_result[n] = u[:, 0]
            x_result[n] = x[:, 0]
            y_result[n] = y[:, 0] 

        return u_result, x_result, y_result
    
