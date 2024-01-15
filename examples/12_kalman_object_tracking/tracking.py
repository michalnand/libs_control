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


if __name__ == "__main__":

    source= cv2.VideoCapture(0)


    noise_var = 0.0005

    #kalman = LibsControl.KalmanFilterUniversal(2, r=noise_var, q=10**-5, mode = "velocity")
    #kalman = LibsControl.KalmanFilterUniversal(2, r=noise_var, q=10**-4, mode = "acceleration")
    #kalman = LibsControl.KalmanFilterACC(2, r=noise_var, q=10**-5)
    kalman = LibsControl.KalmanFilterVel(2, r=noise_var, q=10**-6)


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

        z_now  = pos + (noise_var**0.5)*numpy.random.randn(2)

        #kalman filter step
        z_fil = kalman.step(z_now)
        z_pred = kalman.predict(10)
        
        result_im = frame.copy()


        
  
        x = int(pos[1]*width)
        y = int(pos[0]*height)
        cv2.circle(result_im, (x, y), 40, (0, 1, 0), -1)

        line_gt.append([x, y])
        if len(line_gt) > line_max:
            line_gt = line_gt[1:]


        x = int(z_fil[1]*width)
        y = int(z_fil[0]*height)
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

        z_pred[:, 0]*= height
        z_pred[:, 1]*= width
        z_pred[:, [0, 1]] = z_pred[:, [1, 0]]
        z_pred = z_pred.astype(int)


        line_tmp = numpy.expand_dims(numpy.array(line_gt), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (0, 1, 0), 4)

        line_tmp = numpy.expand_dims(numpy.array(line_noised), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (0, 0, 1), 2)
 
        line_tmp = numpy.expand_dims(numpy.array(line_filter), axis=1)
        result_im = cv2.polylines(result_im, [line_tmp], False, (1, 0, 0), 4)

        #line_tmp = numpy.expand_dims(numpy.array(z_pred), axis=1)
        #result_im = cv2.polylines(result_im, [line_tmp], False, (1, 0, 1), 4)

        

        
        result_im = cv2.putText(result_im, "ground truth", (10, 50), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 1, 0), 4, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "measurement",  (10, 100), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 0, 1), 4, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "filtered",  (10, 150), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (1, 0, 0), 4, cv2.LINE_AA) 
        #result_im = cv2.putText(result_im, "prediction",  (10, 200), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (1, 0, 1), 4, cv2.LINE_AA) 


        result_im = cv2.resize(result_im, (width//2, height//2))

        '''
        if writter is None:
            writter = cv2.VideoWriter('tracking_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width//2, height//2))

        writter.write(numpy.array(255*result_im).astype(numpy.uint8))
        '''
        
        cv2.imshow('frame', result_im)
        if cv2.waitKey(1) == ord('q'):
            break
