import LibsControl
import numpy
import matplotlib.pyplot as plt

import cv2


def detect_object(frame, color_min, color_max, height = 512, width = 512):
    #smaller image
    frame_denoised = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
    frame_denoised = cv2.GaussianBlur(frame_denoised, (5, 5), 1)
    
    hsv = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2HSV)

    mask    = cv2.inRange(hsv, color_min, color_max)
    kernel  = numpy.ones((5, 5))
    mask    = cv2.erode(mask, kernel)
    mask    = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_LINEAR)

    mg      = numpy.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    mask_p  = mask/(mask.sum() + 10**-6)
    mask_p  = numpy.expand_dims(mask_p, 0)

    pos = (mg*mask_p).sum(axis=(1, 2))

    pos[0]/= frame.shape[0]
    pos[1]/= frame.shape[1]

    return pos


'''
class KalmanFilter:
    # Q - process noise covariance, 2x2
    # R - measurement noise covariance, 2x2
    def __init__(self, r):

        self.mat_a = numpy.eye(2)
        self.mat_a[0][1] = 1.0

        self.x_hat = numpy.zeros((2, 1))

        self.p = 0.5*numpy.eye(2)

        self.r = r

    def step(self, x):

        #prediction
        x_pred = self.mat_a @ self.x_hat
        p_pred = self.mat_a @ self.p @ self.mat_a.T

        #update
        y = x - x_pred
        S = p_pred + self.r
        K = p_pred @ numpy.linalg.inv(S)

        #correction
        self.x_hat = x_pred + K @ y
        self.p = p_pred - K  @ p_pred

        return self.x_hat
'''


class KalmanFilter:
    # rx - position noise variance
    # rv - velocity noise variance
    # q  - process noise variance
    def __init__(self, rx, rv, q = 10**-4):

        self.rx = rx
        self.rv = rv
        self.q = q

        self.x_hat = 0.0
        self.v_hat = 0.0

        self.px = 1.0
        self.pv = 1.0

    # x - noised position measurement
    # v - noised velocity measurement
    # returns denoised position and velocity
    def step(self, x, v):
        #state predict
        self.x_hat = self.x_hat + self.v_hat
        self.v_hat = self.v_hat
        self.px = self.px + self.pv
        self.pv = self.pv + self.q

        #kalman gain
        kx = self.px/(self.px + self.rx)
        kv = self.pv/(self.pv + self.rv)

        #update
        self.x_hat = self.x_hat + kx*(x - self.x_hat)
        self.v_hat = self.v_hat + kv*(v - self.v_hat)
        self.px = (1.0 - kx)*self.px
        self.pv = (1.0 - kx)*self.pv

        return self.x_hat, self.v_hat



if __name__ == "__main__":

    source= cv2.VideoCapture(0)


    noise_var = 0.001

    z_now  = numpy.zeros(2)
    z_prev = numpy.zeros(2)

  
    kalman_x = KalmanFilter(noise_var, noise_var, q = 0.0001)
    kalman_y = KalmanFilter(noise_var, noise_var, q = 0.0001)
    
    line_max = 20
    line_gt = []
    line_noised = []
    line_filter = []    

    writter = None


    while True:
        ret, frame = source.read()

        width = frame.shape[1]
        height = frame.shape[0]

        frame = numpy.array(frame/255.0).astype(numpy.float32)

        pos = detect_object(frame, numpy.array([80, 0.2, 0.2]), numpy.array([130, 1.0, 1.0]))

        z_prev = z_now
        z_now  = pos + (noise_var**0.5)*numpy.random.randn(2)

        #estimate velocity
        dz     = z_now - z_prev

        #kalman filter step
        z_fil_x, _ = kalman_x.step(z_now[1], dz[1])

        z_fil_y, _ = kalman_y.step(z_now[0], dz[0])
        
        result_im = frame.copy()


        
  
        x = int(pos[1]*width)
        y = int(pos[0]*height)
        cv2.circle(result_im, (x, y), 40, (0, 1, 0), -1)

        line_gt.append([x, y])
        if len(line_gt) > line_max:
            line_gt = line_gt[1:]


        x = int(z_fil_x*width)
        y = int(z_fil_y*height)
        cv2.circle(result_im, (x, y), 30, (1, 0, 0), -1)

        line_filter.append([x, y])
        if len(line_filter) > line_max:
            line_filter = line_filter[1:]

        x = int(z_now[1]*width)
        y = int(z_now[0]*height)
        cv2.circle(result_im, (x, y), 15, (0, 0, 1), -1)

        line_noised.append([x, y])
        if len(line_noised) > line_max:
            line_noised = line_noised[1:]

        line_tmp = numpy.expand_dims(numpy.array(line_gt), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (0, 1, 0), 4)

        line_tmp = numpy.expand_dims(numpy.array(line_noised), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (0, 0, 1), 2)
 
        line_tmp = numpy.expand_dims(numpy.array(line_filter), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (1, 0, 0), 4)

        
        result_im = cv2.putText(result_im, "ground truth", (10, 50), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 1, 0), 4, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "measurement",  (10, 100), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 0, 1), 4, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "filtered",  (10, 150), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (1, 0, 0), 4, cv2.LINE_AA) 


        result_im = cv2.resize(result_im, (width//2, height//2))

        '''
        if writter is None:
            writter = cv2.VideoWriter('tracking_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width//2, height//2))

        writter.write(numpy.array(255*result_im).astype(numpy.uint8))
        '''
        
        cv2.imshow('frame', result_im)
        if cv2.waitKey(1) == ord('q'):
            break
