import numpy


class MovingSpheres:

    def __init__(self, count, dt = 0.01, x_range = 0.999, v_range = 1.0, a_range = 10.0):
        
        self.count = count
        self.dt = dt

        self.x_range = x_range
        self.v_range = v_range
        self.a_range = a_range

        
        # random position, velocity and acceleration
        self.a = self.a_range*(2.0*numpy.random.rand(count, 2) - 1.0)
        self.v = self.v_range*(2.0*numpy.random.rand(count, 2) - 1.0)
        self.x = self.x_range*(2.0*numpy.random.rand(count, 2) - 1.0)

        self.a = numpy.clip(self.a, -self.a_range, self.a_range)
        self.v = numpy.clip(self.v, -self.v_range, self.v_range)
        self.x = numpy.clip(self.x, -self.x_range, self.x_range)



    def step(self): 

        wall_distance = 0.005
        

        speed = 2.0

        #distances from boudaries
        d_top  = (self.x[:, 1] - (-1.0))**2
        idx = numpy.where(d_top < wall_distance)[0]
        self.v[idx, 1] = self.v_range*numpy.random.rand()

        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_bottom  = (self.x[:, 1] - 1.0)**2
        idx = numpy.where(d_bottom < wall_distance)[0]
        self.v[idx, 1] = -self.v_range*numpy.random.rand()
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_right  = (self.x[:, 0] - 1.0)**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = -self.v_range*numpy.random.rand()
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_right  = (self.x[:, 0] - (-1.0))**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = self.v_range*numpy.random.rand()   
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)

        # motion equations
        self.a = self.a
        self.v = self.v + self.a*self.dt
        self.x = self.x + self.v*self.dt

        # limit range
        self.a = numpy.clip(self.a, -self.a_range, self.a_range)
        self.v = numpy.clip(self.v, -self.v_range, self.v_range)
        self.x = numpy.clip(self.x, -self.x_range, self.x_range)



        return self.x






class Fireworks:

    def __init__(self, count, dt = 0.01, x_range = 0.999, v_range = 1.0, a_range = 10.0):
        
        self.count = count
        self.dt = dt

        self.x_range = x_range
        self.v_range = v_range
        self.a_range = a_range

        
        # random position, velocity and acceleration
        self.reset()

        self.steps = 0


    def reset(self):
        # random position, velocity and acceleration
        self.a = numpy.zeros((self.count, 2))
        self.v = numpy.zeros((self.count, 2))
        self.x = 0.01*self.x_range*(2.0*numpy.random.rand(self.count, 2) - 1.0)

        self.v[:, 1] = -1.0

        self.x[:, 0]+= 0.05
        self.x[:, 1]+= 0.9

        self.a = numpy.clip(self.a, -self.a_range, self.a_range)
        self.v = numpy.clip(self.v, -self.v_range, self.v_range)
        self.x = numpy.clip(self.x, -self.x_range, self.x_range)



    def step(self): 

        self.steps+= 1


        if self.steps == 100:
            self.a = self.a_range*(2.0*numpy.random.rand(self.count, 2) - 1.0)
            #self.v = self.v_range*(2.0*numpy.random.rand(self.count, 2) - 1.0) 

        if self.steps >= 500:
            self.reset()
            self.steps = 0

        
            

        wall_distance = 0.005
        

        #distances from boudaries
        d_top  = (self.x[:, 1] - (-1.0))**2
        idx = numpy.where(d_top < wall_distance)[0]
        self.v[idx, 1] = self.v_range*numpy.random.rand()

        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_bottom  = (self.x[:, 1] - 1.0)**2
        idx = numpy.where(d_bottom < wall_distance)[0]
        self.v[idx, 1] = -self.v_range*numpy.random.rand()
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_right  = (self.x[:, 0] - 1.0)**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = -self.v_range*numpy.random.rand()
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)


        d_right  = (self.x[:, 0] - (-1.0))**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = self.v_range*numpy.random.rand()   
        self.a[idx] = self.a_range*(2.0*numpy.random.rand(2) - 1.0)

        # motion equations
        self.a = self.a
        self.v = self.v + self.a*self.dt
        self.x = self.x + self.v*self.dt

        # limit range
        self.a = numpy.clip(self.a, -self.a_range, self.a_range)
        self.v = numpy.clip(self.v, -self.v_range, self.v_range)
        self.x = numpy.clip(self.x, -self.x_range, self.x_range)



        return self.x


