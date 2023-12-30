import torch
import numpy
import cv2

import LibsControl

 
class RubberBot(torch.nn.Module):
    def __init__(self, m = 0.5, c = 0.7, k = 1.9, width = 10, height = 5, fully_observed = True):
        super().__init__()

        points_count = width*height

        self.am      = torch.zeros((points_count, points_count), dtype=int)

        for j in range(points_count):
            for i in range(points_count):
                ya = (i//width)
                xa = (i%width)

                yb = (j//width) 
                xb = (j%width)

                if abs(xa - xb) + abs(ya - yb) == 1:
                    self.am[j][i] = 1
                    self.am[i][j] = -1


        print(self.am)

        
        self.mat_a = torch.zeros((2*2*points_count, 2*2*points_count))
        self.mat_b = torch.zeros((4, 1))
        


        self.mat_a[0][2] =  1.0
        self.mat_a[1][3] =  1.0

        self.mat_a[2][0] =  -k/m1
        self.mat_a[2][1] =   k/m1
        self.mat_a[2][2] = -c1/m1

        self.mat_a[3][0] =   k/m2
        self.mat_a[3][1] =   -k/m2
        self.mat_a[3][3] = -c2/m2


        self.mat_b[2][0] = 1.0/m1

        

        if fully_observed:
            self.mat_c = torch.zeros((4, 4))
            self.mat_c[0][0] = 1.0
            self.mat_c[1][1] = 1.0
            self.mat_c[2][2] = 1.0 
            self.mat_c[3][3] = 1.0
        else:
            self.mat_c = torch.zeros((4, 2))
            self.mat_c[0][0] = 1.0
            self.mat_c[1][1] = 1.0
            

        self.plant       = LibsControll.DynamicalSystem(self.mat_a, self.mat_b, self.mat_c)

    def reset(self):
        self.plant.reset()

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




if __name__ == "__main__":
    
    plant = RubberBot(width = 4, height = 4)