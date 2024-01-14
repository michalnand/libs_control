import LibsControl
import numpy
import cv2

class BumpingSpheres:

    def __init__(self, count, dt = 0.01):
        
        self.count = count
        self.dt = dt

        self.x = numpy.zeros((count, 2))

        self.x[:, 0] = 2.0*numpy.random.rand(count) - 1.0
        self.x[:, 1] = -0.9 + 0.1*numpy.random.randn(count)

        self.v = numpy.random.randn(count, 2)
        self.a = numpy.zeros((count, 2))

        self.x = numpy.clip(self.x, -0.99, 0.99)


    def step(self): 

        eps = 10**-8

        wall_distance = 0.001
        wall_rf = 0.002
        sphere_rf = 0.5

        g_force = 4.0

        '''
        #each by each distance
        d       = numpy.expand_dims(self.x, 0) - numpy.expand_dims(self.x, 1)
        d_norm  = (d**2).sum(axis=-1)
        d_norm  = numpy.expand_dims(d_norm, 2)

        a = sphere_rf*d/(d_norm + eps)
        self.a = a.sum(axis=0)
        '''

        self.a[:, 1] = 4

        #distances from boudaries

        d_bottom  = (self.x[:, 1] - 1.0)**2
        idx = numpy.where(d_bottom < wall_distance)[0]
        self.v[idx, 1] = -g_force*0.9


        d_right  = (self.x[:, 0] - 1.0)**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = -g_force*0.1


        d_right  = (self.x[:, 0] - (-1.0))**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.v[idx, 0] = g_force*0.1
         

        '''
        d_right  = (self.x[:, 0] - 1.0)**2
        idx = numpy.where(d_right < wall_distance)[0]
        self.a[idx, 0] = -wall_rf/(d_right[idx] + eps)

        d_left  = (self.x[:, 0] - (-1.0))**2
        idx = numpy.where(d_left < wall_distance)[0]
        self.a[idx, 0] = wall_rf/(d_left[idx] + eps)

        
        d_bottom  = (self.x[:, 1] - 1.0)**2
        idx = numpy.where(d_bottom < wall_distance)[0]
        self.a[idx, 1] = -wall_rf/(d_bottom[idx] + eps)

        d_top  = (self.x[:, 1] - (-1.0))**2
        idx = numpy.where(d_top < wall_distance)[0]
        self.a[idx, 1] = wall_rf/(d_top[idx] + eps)
        '''
     

        self.v = self.v + self.a*self.dt
        self.x = self.x + self.v*self.dt


        self.x = numpy.clip(self.x, -0.9999, 0.9999)

        return self.x



def plot_trajectory(result_im, trajectory, color):
    line_tmp = numpy.array(numpy.expand_dims(trajectory, axis=1)).astype(int)
    result_im = cv2.polylines(result_im, [line_tmp], False, color, 1)

    for j in range(0, line_tmp.shape[0], 4):
        cv2.circle(result_im, (line_tmp[j, 0, 0], line_tmp[j, 0, 1]), 2, color, -1)

    return result_im


if __name__ == "__main__":


    n_count = 3
    prediction_steps = 32

    noise_variance = 0.00001

    bs = BumpingSpheres(n_count)

    colors = numpy.random.rand(n_count, 3)

    colors[:] = 1
    
    #kalman_v   = LibsControl.KalmanFilter(n_count*2, noise_variance)
    #kalman_acc = LibsControl.KalmanFilterACC(n_count*2, noise_variance)
    kalman_v   = LibsControl.KalmanFilterUniversal(n_count*2, r=noise_variance, q=10**-8, mode = "velocity")
    kalman_acc = LibsControl.KalmanFilterUniversal(n_count*2, r=noise_variance, q=10**-8, mode = "acceleration")

    height = 512
    width  = 1024    


    writter = None

    while True:

        x_pos = bs.step()


        x_noised = x_pos + (noise_variance)*numpy.random.randn(x_pos.shape[0], x_pos.shape[1])

        kalman_v.step(x_noised.reshape(n_count*2))
        kalman_acc.step(x_noised.reshape(n_count*2))


        x_pred_v = kalman_v.predict(prediction_steps)
        x_pred_v = x_pred_v.reshape((prediction_steps, n_count, 2))
        x_pred_v[:, :, 0] = 0.5*(x_pred_v[:, :, 0] + 1.0)*width
        x_pred_v[:, :, 1] = 0.5*(x_pred_v[:, :, 1] + 1.0)*height

        x_pred_acc = kalman_acc.predict(prediction_steps)
        x_pred_acc = x_pred_acc.reshape((prediction_steps, n_count, 2))
        x_pred_acc[:, :, 0] = 0.5*(x_pred_acc[:, :, 0] + 1.0)*width
        x_pred_acc[:, :, 1] = 0.5*(x_pred_acc[:, :, 1] + 1.0)*height



        result_im = numpy.zeros((height, width, 3), dtype=numpy.float32)

        for i in range(n_count):

            x = int((x_pos[i][0] + 1.0)*0.5*width)
            y = int((x_pos[i][1] + 1.0)*0.5*height) 
            c = colors[i]

            result_im = plot_trajectory(result_im, x_pred_v[:, i, :], (1, 0, 0))
            result_im = plot_trajectory(result_im, x_pred_acc[:, i, :], (0, 1, 0))
            
          
            cv2.circle(result_im, (x, y), 10, c, -1)


            

        result_im = cv2.putText(result_im, "current position", (10, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, colors[0], 2, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "kalman velocity model prediction", (10, 80), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (1, 0, 0), 2, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "kalman acc model prediction", (10, 110), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 1, 0), 2, cv2.LINE_AA) 

        cv2.imshow("visualisation", result_im)
        cv2.waitKey(1)

        '''
        if writter is None:
            writter = cv2.VideoWriter('spheres_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

        writter.write(numpy.array(255*result_im).astype(numpy.uint8))
        '''