import numpy
import LibsControl

import cv2

class RocketLanding(LibsControl.DynamicalSystem):
    def __init__(self, dt):
        M = 549.0*1000   # mass of rocket,  kg
        L = 70.0         # height of rocket, m


        #state x : x, y, vx, vy, theta, dtheta, alpha, dalpha

        mat_a = numpy.zeros((8, 8))
        mat_b = numpy.zeros((8, 2))
        mat_c = numpy.zeros((8, 8)) 

        for i in range(mat_a.shape[0]):
            mat_a[i][i] = 1.0

        mat_a[0][2] = dt
        mat_a[1][3] = dt
        mat_a[2][4] = dt/M
        mat_a[2][6] = dt/M

        mat_a[4][5] = dt
        mat_a[5][6] = -12.0*dt/(M*L)

        mat_a[6][7] = dt


        mat_b[0][2] = dt/M
        mat_b[0][3] = dt/M
        mat_b[0][5] = -12.0*dt/(M*L)
        mat_b[1][7] = dt

        for i in range(mat_c.shape[0]):
            mat_c[i][i] = 1.0


        LibsControl.DynamicalSystem.__init__(self, mat_a, mat_b, mat_c, dt)

       

    def render(self, width = 800, height = 500, scale = 0.01):
        # rocket position
        x_pos = self.x[0][0]    
        y_pos = self.x[1][0]

        # rocket angle
        theta = self.x[4][0]

        # nozzle angle
        alpha = self.x[6][0]


        result_img = numpy.zeros((width, height, 3))

