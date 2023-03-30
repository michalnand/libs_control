import torch
import numpy
import cv2

import LibsControl


class BalancingRobot(torch.nn.Module):
    def __init__(self, dt = 0.001):
        super().__init__()

        self.dt = dt

        M = 0.5;        # Mass of carriage (kg)
        m = 0.95;       # Mass of pendulum (kg)
        b = 0.1;        # Drag Coefficient (N/m/s)
        J = 0.0076;     # Pendulum Moment of Inertia (kg.m^2)
        g = 9.8;        # Gravity acceleration (m/s^2)
        l = 0.155;      # Half-length of pendulum (m)


        self.mat_a = torch.zeros((4, 4))
        self.mat_b = torch.zeros((4, 1))
        self.mat_c = torch.zeros((4, 4))

        den = J*(M + m)+M*m*(l**2)


        self.mat_a[0][1] =  1.0
        self.mat_a[1][1] =  -(J + m*(l**2))*b/den
        self.mat_a[1][2] = (m**2)*g*(l**2)/den

        self.mat_a[2][3] =  1.0
        self.mat_a[3][1] = -m*b*l/den
        self.mat_a[3][2] = m*g*l*(M + m)/den


        self.mat_b[1][0] = (J + m*(l**2))/den
        self.mat_b[3][0] = (m*l)/den
        

        self.mat_c[0][0] = 1.0
        self.mat_c[1][1] = 1.0
        self.mat_c[2][2] = 1.0 
        self.mat_c[3][3] = 1.0



    def forward(self, x, u):
        dx = self.mat_a@x + self.mat_b@u

        x  = x + dx*self.dt
        y  = self.mat_c@x
        
        return y

    
    #TODO
    def render(self):
        height    = 256
        width     = 512

        min_range = -4.0
        max_range = 4.0

        x1r = float(self.x[0][0].detach().to("cpu").numpy())
        x2r = float(self.x[0][1].detach().to("cpu").numpy())

        x1 = numpy.clip(x1r, min_range*0.999, max_range*0.999)
        x2 = numpy.clip(x1r + x2r, min_range*0.999, max_range*0.999)

        #print("rendering ", x1r, x2r)

        x1 = int(width*(x1 - min_range)/(max_range - min_range))
        x2 = int(width*(x2 - min_range)/(max_range - min_range))

        y1 = height//2
        y2 = height//2
        w  = 60
        h  = 40


        image = numpy.zeros((height, width, 3))



        image = cv2.putText(image, "position 1 = " + str(round(x1r, 2)) + "[m]", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.putText(image, "position 2 = " + str(round(x2r, 2)) + "[m]", (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


        image = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        image = cv2.rectangle(image, (x1 - w//2, y1 - h//2), (x1 + w//2, y1 + h//2), (255, 0, 0), -1)
        image = cv2.rectangle(image, (x2 - w//2, y2 - h//2), (x2 + w//2, y2 + h//2), (0, 0, 255), -1)

        image = cv2.putText(image, "1", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, "2", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Displaying the image 
        cv2.imshow("dynamical system", image) 
        cv2.waitKey(1)

