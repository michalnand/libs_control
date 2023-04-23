import numpy

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw

from obj_model import *

class Render:
    def __init__(self, height, width, model_file_name = "robot_model/robot"):        
        self.model = ObjModel(model_file_name)

        self.width = width
        self.height = height

        res = self._create_window(self.height, self.width)
        print("init res = ", res)
        
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
        
        return 0


    def render(self, x_pos, y_pos, phi, theta, scale = 0.005):

        aspect = self.width/self.height
        glViewport(0, 10, 2*self.width, 2*self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
         

        glEnable(GL_DEPTH_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glClearColor(0, 0, 0, 0)
        
        glPushMatrix()
        glTranslatef(x_pos, 0, y_pos)

        self.rotate(0, 0, theta)

        
        for i in range(len(self.model.polygons)):
            color = self.model.colors[i]
            
            self.paint_polygon(self.model.points, self.model.polygons[i], color, scale)
    

        glPopMatrix()


        glfw.swap_buffers(self.window)
        glfw.poll_events()


    def paint_polygon(self, points, polygon, color, scale = 1.0):

        glBegin(GL_TRIANGLES)

        glColor3f(color[0], color[1], color[2])

        for i in range(len(polygon)):
            idx = polygon[i]
            glVertex3f(points[idx][1]*scale, points[idx][2]*scale, points[idx][2]*scale)
        
        glEnd()


    def rotate(self, angle_x, angle_y, angle_z):
        glRotatef(angle_x, 1.0, 0.0, 0.0)
        glRotatef(angle_y, 0.0, 1.0, 0.0)
        glRotatef(angle_z, 0.0, 0.0, 1.0)

