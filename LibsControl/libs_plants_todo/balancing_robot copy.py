import torch
import numpy
import cv2

import LibsControl


class BalancingRobot(torch.nn.Module):
    def __init__(self):
        super().__init__()


        g   = 9.81      #gravitational acceleration [m/s^2]
        mb  = 2.63      #mass of robot [kg]
        jb  = 0.018     #moment of inertia about CoM, [kgm^2]
        r   = 0.05      #radius of wheels [m]

        mw  = 0.14      #mass of wheel [kg]         
        jw  = 0.000175  #moment of inertia of wheels, [kgm^2]
        l   = 0.18      #length measured from the wheel axis and mass center of the main body [m]

        mu0 = 0.1       #coeficient of friction wheel and ground
        mu1 = 0.0       #coeficient of friction chassis and wheel


        self.mat_a = torch.zeros((4, 4))
        self.mat_b = torch.zeros((4, 1))
        self.mat_c = torch.zeros((4, 4))

        den = (mb + 2*mw + 2*jw/(r**2))*(mb*(l**2) + jb) - ((mb*l)**2)


        self.mat_a[0][2] =  1.0
        self.mat_a[1][3] =  1.0
        
        self.mat_a[2][1] = -((mb*l)**2)*g/den
        self.mat_a[2][2] = -2*mu0*(mb*(l**2) + jb)/den
        self.mat_a[2][3] = 2*mb*l*mu1/den

        self.mat_a[3][1] = (mb + 2*mw + 2*jw/(r**2))*(mb*g*l)/den
        self.mat_a[3][2] = 2*mu0*mb*l/den
        self.mat_a[3][3] = -2*mu1*(mb + 2*mw + 2*jw/(r**2))/den
      
        self.mat_b[2][0] = -mb*l/den
        self.mat_b[3][0] = (mb+2*mw+2*jw/(r**2))/den

      
        self.mat_c[0][0] = 1.0
        self.mat_c[1][1] = 1.0
        self.mat_c[2][2] = 1.0 
        self.mat_c[3][3] = 1.0

        self.plant       = LibsControll.DynamicalSystem(self.mat_a, self.mat_b, self.mat_c)



    def forward(self, x, u):
        self.x = x
        return self.plant.forward(x, u)

    def __repr__(self):
        return str(self.plant)


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




'''
class BalancingRobot(torch.nn.Module):
    def __init__(self):
        super().__init__()


        mb  = 0.595     #mass of robot [kg]
        mw  = 0.031     #mass of wheel [kg]         
        jb  = 0.0015    #moment of inertia about CoM, [kgm^2]
        r   = 0.04      #radius of wheels [m]
        jw  = 0.0000596 #moment of inertia of wheels, [kgm^2]
        l   = 0.08      #length measured from the wheel axis and mass center of the main body [m]
        ke  = 0.468     #EMF motor constant [Vs/rad]
        km  = 0.317     #motor constant [Nm/A]
        res = 6.69      #motor resistance [ohm]
        b   = 0.002     #viscous friction constant [Nms/rad]
        g   = 9.81      #gravitational acceleration [m/s^2]


        tmp     = res*( 2.0*(jb*jw + jw*(l**2)*mb + jb*mw*(r**2) + (l**2)*mb*mw*(r**2) ) + jb*mb*(r**2) )
        alpha   = 2*(r*b - ke*km)*(mb*(l**2) + mb*r*l + jb)/tmp

        tmp     = jb*(2*jw + mb*(r**2) + 2*mw*(r**2)) + 2*jw*(l**2)*mb + 2*(l**2)*mb*mw*(r**2)
        beta    = -(l**2)*(mb**2)*g*(r**2)/tmp

        tmp     = res*r*(2.0*(jb*jw + jw*(l**2)*mb + jb*mw*(r**2) + (l**2)*mb*mw*(r**2)) + jb*mb*(r**2))
        gamma   = -2.0*(res*b - ke*km)*(2*jw + mb*(r**2) + 2*mw*(r**2) + l*mb*r)/tmp

        tmp     = 2*jb*jw + 2*jw*(l**2)*mb + jb*mb*(r**2) + 2*jb*mw*(r**2) + 2*(l**2)*mb*mw*(r**2)
        delta   = l*mb*g*(2*jw + mb*(r**2) + 2*mw*(r**2))/tmp

        epsilon = (km*r)/(res*b - ke*km)

        self.mat_a = torch.zeros((4, 4))
        self.mat_b = torch.zeros((4, 1))
        self.mat_c = torch.zeros((4, 4))


        self.mat_a[0][1] =  1.0
        
        self.mat_a[1][1] = alpha
        self.mat_a[1][2] = beta
        self.mat_a[1][3] = -r*alpha
        
        self.mat_a[2][3] =  1.0

        self.mat_a[3][1] = gamma
        self.mat_a[3][2] = delta
        self.mat_a[3][3] = -r*gamma
        

        self.mat_b[1][0] = alpha*epsilon
        self.mat_b[3][0] = gamma*epsilon

      
        self.mat_c[0][0] = 1.0
        self.mat_c[1][1] = 1.0
        self.mat_c[2][2] = 1.0 
        self.mat_c[3][3] = 1.0

        self.plant       = LibsControll.DynamicalSystem(self.mat_a, self.mat_b, self.mat_c)



    def forward(self, x, u):
        self.x = x
        return self.plant.forward(x, u)

    def __repr__(self):
        return str(self.plant)


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

'''