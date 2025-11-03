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

class BalancingRobot(LibsControl.DynamicalSystem):
    def __init__(self, dt):

        M = 0.5;        # Mass of carriage (kg)
        m = 0.95;       # Mass of pendulum (kg)
        b = 0.1;        # Drag Coefficient (N/m/s)
        J = 0.0076;     # Pendulum Moment of Inertia (kg.m^2)
        Jb= 0.005       # body inertia, along vertical axis (kg.m^2)
        g = 9.8;        # Gravity acceleration (m/s^2)
        l = 0.155;      # Half-length of pendulum (m)
        L = 0.08        # half length of wheel to wheel distance (m)


        mat_a = numpy.zeros((6, 6))
        mat_b = numpy.zeros((6, 2))
        mat_c = numpy.zeros((3, 6)) 

        den = J*(M + m) + M*m*(l**2) 


        mat_a[0][1] =  1.0
        mat_a[1][1] =  -(J + m*(l**2))*b/den
        mat_a[1][2] = (m**2)*g*(l**2)/den

        mat_a[2][3] =  1.0
        mat_a[3][1] = -m*b*l/den
        mat_a[3][2] = m*g*l*(M + m)/den


        mat_a[4][5] =  1.0
        mat_a[5][5] =  -20.0
        

        mat_b[1][0] = (J + m*(l**2))/den
        mat_b[3][0] = (m*l)/den
        mat_b[1][1] = (J + m*(l**2))/den
        mat_b[3][1] = (m*l)/den

        mat_b[5][0] =  L/Jb
        mat_b[5][1] = -L/Jb
    

        mat_c[0][0] = 1.0
        mat_c[1][2] = 1.0
        mat_c[2][4] = 1.0

        LibsControl.DynamicalSystem.__init__(self, mat_a, mat_b, mat_c, dt)

        #for rendring
        self.window = None

    def render(self):
        r = 0.6

        if self.window is None:
            self._create_window(512, 512)

        x_pos = self.y[0][0]
        y_pos = 0
        theta = self.y[1][0]*180.0/numpy.pi
        phi   = self.y[2][0]*180.0/numpy.pi

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 
        glLoadIdentity() 

       
        glClearColor(0.18, 0.09, 0.2, 0)
        glTranslatef(0.0, 0.0, -3.0)

        self.rotate(20, 20, 0)

        
        grid_size = 20
        d = 1/(grid_size)
        glPushMatrix()

        
        for j in range(grid_size):
            for i in range(grid_size):

                glPushMatrix()

                if (j%2) == (i%2):
                    glColor3f(0.3, 0.3, 0.3)
                else:
                    glColor3f(0.1, 0.1, 0.1)
                    

                x = 2.0*(j/grid_size - 0.5)
                y = 2.0*(i/grid_size - 0.5)

                glTranslatef(x, 0, y)

                glBegin(GL_QUADS)
                glVertex3f(d, 0, -d)
                glVertex3f(-d, 0, -d)
                glVertex3f(-d, 0, d)
                glVertex3f(d, 0, d)
                glEnd()

                glPopMatrix()
        

        glPopMatrix()


        glPushMatrix()



        w = 0.04
        h = 0.4
        d = 0.3
        r = 0.05

        glTranslatef(x_pos, r, y_pos)
        self.rotate(0, phi, theta)

        glPushMatrix()
        glTranslatef(0, h/2, 0)
        glColor3f(0.11, 0.56, 1.0)
        self.paint_cuboid(w, h, d)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0, d/2 + 0.01)
        glColor3f(0.4, 0.0, 1.0)
        self.paint_circle(r)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0, -d/2 - 0.01)
        glColor3f(0.4, 0.0, 1.0)
        self.paint_circle(r)
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


    def rotate(self, angle_x, angle_y, angle_z):
        glRotatef(angle_x, 1.0, 0.0, 0.0)
        glRotatef(angle_y, 0.0, 1.0, 0.0)
        glRotatef(angle_z, 0.0, 0.0, 1.0)

   
    def paint_cuboid(self, width, height, depth):
        w = width/2.0
        h = height/2.0
        d = depth/2.0

        glBegin(GL_QUADS);        

        glVertex3f( w, h, -d)
        glVertex3f(-w, h, -d)
        glVertex3f(-w, h,  d)
        glVertex3f( w, h,  d)

        glVertex3f( w, -h,  d)
        glVertex3f(-w, -h,  d)
        glVertex3f(-w, -h, -d)
        glVertex3f( w, -h, -d)

        glVertex3f( w,  h, d)
        glVertex3f(-w,  h, d)
        glVertex3f(-w, -h, d)
        glVertex3f( w, -h, d)

        glVertex3f( w, -h, -d)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w,  h, -d)
        glVertex3f( w,  h, -d)

        glVertex3f(-w,  h,  d)
        glVertex3f(-w,  h, -d)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, -h,  d)

        glVertex3f(w,  h, -d)
        glVertex3f(w,  h,  d)
        glVertex3f(w, -h,  d)
        glVertex3f(w, -h, -d)

        glEnd()

    def paint_circle(self, radius, steps = 32):
        pi = 3.141592654
        glBegin(GL_POLYGON)
        
        for i in range(steps):
            angle = i*2.0*pi/steps
            glVertex3f(radius*numpy.cos(angle), radius*numpy.sin(angle), 0.0)
        
        glEnd()