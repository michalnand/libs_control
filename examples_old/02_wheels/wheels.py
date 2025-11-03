import numpy
import LibsControl


#for rendering
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import time


#three wheels connected with springs and controlled with two motors
#x state = (pos0, pos1, pos2, vel0, vel1, vel2)

class Wheels(LibsControl.DynamicalSystem):
    def __init__(self, dt):

        c0 = 2.697*1e-3
        c1 = 2.66 *1e-3
        c2 = 3.05 *1e-3
        c3 = 2.85 *1e-3

        d0 = 6.78*1e-5
        d1 = 8.01*1e-5
        d2 = 8.82*1e-5

        J  = 2.25e-4

        mat_a = numpy.zeros((6, 6))
        mat_b = numpy.zeros((6, 2))
        mat_c = numpy.zeros((3, 6))


            
        mat_a[0][3] =  1.0
        mat_a[1][4] =  1.0
        mat_a[2][5] =  1.0

        mat_a[3][0] =  (-c0-c1)/J
        mat_a[3][1] =   c1/J
        mat_a[3][3] =  -d0/J

        mat_a[4][0] =  c1/J
        mat_a[4][1] =  (-c1-c2)/J
        mat_a[4][2] =  c2/J
        mat_a[4][4] =  -d1/J

        mat_a[5][1] =   c2/J
        mat_a[5][2] =   -c2/J
        mat_a[5][3] =   -c3/J
        mat_a[5][5] =  -d2/J

        mat_b[3][0] = c0/J
        mat_b[5][1] = c3/J

        #observing only position
        mat_c[0][0] = 1.0
        mat_c[1][1] = 1.0
        mat_c[2][2] = 1.0


        LibsControl.DynamicalSystem.__init__(self, mat_a, mat_b, mat_c, dt)

        #for rendring
        self.window = None

    def render(self):

        r = 0.6

        if self.window is None:
            self._create_window(512, 512)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 
        glLoadIdentity() 

       
        glClearColor(0.18, 0.09, 0.2, 0)
        glTranslatef(0.0, 0.0, -4.0)

        glRotatef(-40, 0.0, 1.0, 0.0)


        glPushMatrix()


        glPushMatrix()
        glTranslatef(0, 0, -0.5)
        angle = self.y[0, 0]*180.0/numpy.pi
        glRotatef(angle, 0.0, 0.0, 1.0)

        glColor3f(1.0, 0.0, 0.0)
        self._paint_circle(r)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0, 0)
        angle = self.y[1, 0]*180.0/numpy.pi
        glRotatef(angle, 0.0, 0.0, 1.0)

        glColor3f(0.0, 1.0, 0.0)
        self._paint_circle(r)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0, 0.5)
        angle = self.y[2, 0]*180.0/numpy.pi
        glRotatef(angle, 0.0, 0.0, 1.0)

        glColor3f(0.0, 0.0, 1.0)
        self._paint_circle(r)
        glPopMatrix()


        glPopMatrix()


        
        glfw.swap_buffers(self.window)
        glfw.poll_events()


    def _create_window(self, height, width, label = "visualisation"):

        # Initialize the library
        if not glfw.init():
            return -1
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(width, height, label, None, None)
        if not self.window:
            glfw.terminate()
            return -2

        
        # Make the window's context current
        glfw.make_context_current(self.window)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(100.0) 
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)   
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        return 0

    def _paint_circle(self, radius, steps = 32):
        pi = 3.141592654
        glBegin(GL_POLYGON)
        
        for i in range(steps):
            angle = i*2.0*pi/steps

            if i == 1:
                r = 1.2*radius
            else:
                r = radius
            glVertex3f(r*numpy.cos(angle), r*numpy.sin(angle), 0.0)
        
        glEnd()