import numpy
import LibsControl
import cv2


class RobotTest:

    def __init__(self, controller, robot_model):
        self.controller     = controller
        self.robot_model    = robot_model

        self.max_x = 1.0
        self.max_y = 1.0
        
        self.target_x, self.target_y = self._get_target()
        

        self.x_hat = numpy.zeros((self.robot_model.x.shape[0], 1))
        self.u     = numpy.zeros((self.robot_model.ds.mat_b.shape[1], 1))

    def step(self, eps = 0.001):

        x_r, theta_r = self._get_required(self.target_x, self.target_y)

        #create required state
        xr = self.robot_model.x.copy()
        
        #first rotate robot to given direction
        if abs(theta_r - self.robot_model.theta) > eps:
            xr[3] = theta_r
        #move robot forward
        elif abs(x_r) > eps:
            xr[2] = x_r
            xr[3] = theta_r
        else:
            self.target_x, self.target_y = self._get_target()

        y = self.robot_model.get_y()
        self.x_hat, self.u = self.controller.forward(self.x_hat, self.u, xr, y)

        self.robot_model.step(self.u)

        print(">>> ", self.target_x, self.target_y, self.robot_model.x_pos, self.robot_model.y_pos)
         
    

    def _get_required(self, x_pos_req, y_pos_req):

        dx = x_pos_req - self.robot_model.x_pos
        dy = y_pos_req - self.robot_model.y_pos

        ds      = numpy.sqrt(dx**2 + dy**2)
        dtheta  = numpy.arctan2(dx, dy)

        y       = self.robot_model.get_y()
        x       = y[2] + ds
        theta   = y[3] + dtheta

        return x, theta
    

    def _get_target(self):
        x = self.max_x*(2.0*numpy.random.rand() - 1.0)
        y = self.max_y*(2.0*numpy.random.rand() - 1.0)

        return x, y
        