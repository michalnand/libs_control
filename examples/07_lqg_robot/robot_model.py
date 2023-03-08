import numpy

import LibsControl
import cv2

class RobotModel(LibsControl.DynamicalSystem): 

    '''
        tau1 - robot time constant  [s]
        tau2 - robot time constant  [s]
        rpm - motor free run speed [rpm], when controll input is 1.0 (maximum)
        r   - wheel radius [mm]
        l   - wheels distance [mm]


        model : 

        state x = (velocity, angular rate, distance, angle)

        dv = -1.0/tau1 + 0.5*(b/tau1)*r+(ur + ul)
        dw = -1.0/tau2 + (b/tau2)*(r/2)*(ur - ul)
        dx = v
        da = w
    '''
    def __init__(self, tau1 = 0.3, tau2 = 0.2, rpm = 1000.0, r = 25.0, l = 80.0, dt = 1.0/256.0, pos_range = 4.0, render_size = 400):
        b   = rpm*((2.0*numpy.pi)/60.0) #free run motor velocity, RPM to rad/s
    
        self.r   = r*0.001 #wheel  radius to meters
        self.l   = l*0.001 #wheels distance to meters

        mat_a = numpy.zeros((4, 4)) 
        mat_b = numpy.zeros((4, 2))
        mat_c = numpy.zeros((4, 4))

        #create state space model
        mat_a[0][0] = -1.0/tau1 
        mat_a[1][1] = -1.0/tau2
        mat_a[2][0] = 1.0
        mat_a[3][1] = 1.0


        mat_b[0][0] = 0.5*(b/tau1)*self.r
        mat_b[0][1] = 0.5*(b/tau1)*self.r

        mat_b[1][0] = (b/tau2)*(self.r/self.l) 
        mat_b[1][1] = -(b/tau2)*(self.r/self.l)

        mat_c[0][0] = 1.0
        mat_c[1][1] = 1.0
        mat_c[2][2] = 1.0
        mat_c[3][3] = 1.0


        super().__init__(mat_a, mat_b, mat_c, dt)

        self.pos_range = pos_range
        self.render_size = render_size

        self.reset()


    def reset(self, x_pos = 0.0, y_pos = 0.0, theta = 0.0):

        self.x_pos      = x_pos
        self.y_pos      = y_pos
        self.theta      = theta

        self.x          = numpy.zeros((self.mat_a.shape[0], 1))


    def step(self, u):
        self.x, y = self.forward(self.x, u)

        ds      = y[0]
        dtheta  = y[1]

        self.theta+= dtheta

        self.x_pos+= ds*numpy.cos(self.theta)
        self.y_pos+= ds*numpy.sin(self.theta)

        return self.x_pos, self.y_pos, self.theta
    

    def render(self):

        img = numpy.zeros((self.render_size, self.render_size, 3)) 


        x_pos = self._convert_pos(self.x_pos)
        y_pos = self._convert_pos(self.y_pos)
        l     = self._convert_pos(self.l)
        r     = self._convert_pos(self.r)

        #img = self._draw_angled_rec(img, x_pos, y_pos, l, r, self.theta)

        print(">>> ", self.l,  self.r, l, r)

        cv2.imshow('visualisation', img)
        cv2.waitKey(1)

        return img
    

    def _convert_pos(self, x, scale = 0.01):
        y =  ((x*scale/self.pos_range) + 1.0)*0.5

        print(x, y)

        #y =  y*self.render_size

        #y =  numpy.clip(y, 0, self.render_size)

        return y

    

    def _draw_angled_rec(self, img, x0, y0, width, height, angle):

        #_angle = angle * numpy.pi / 180.0
        b = numpy.cos(angle) * 0.5
        a = numpy.sin(angle) * 0.5
        pt0 = (int(x0 - a * height - b * width),
            int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width),
            int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        cv2.line(img, pt0, pt1, (1.0, 1.0, 1.0), 2)
        cv2.line(img, pt1, pt2, (1.0, 1.0, 1.0), 2)
        cv2.line(img, pt2, pt3, (1.0, 1.0, 1.0), 2)
        cv2.line(img, pt3, pt0, (1.0, 1.0, 1.0), 2)

        return img

    '''
    def get_y(self):
        return self.ds.mat_c@self.x
    
    def get_required_state(self, x_pos_req, y_pos_req):

        dx = x_pos_req - self.x_pos
        dy = y_pos_req - self.y_pos

        ds      = numpy.sqrt(dx**2 + dy**2)
        dtheta  = numpy.arctan2(dx, dy)

        xr      = numpy.zeros((self.ds.mat_a.shape[0], 1))

        xr[2]   = self.x[2] + ds
        xr[3]   = self.x[3] + dtheta
        
        return xr
    '''

