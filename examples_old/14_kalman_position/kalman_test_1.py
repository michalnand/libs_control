import numpy

import tracking_kalman_filter

import numpy
import cv2

import moving_objects



def plot_trajectory(result_im, trajectory, color):
    # Ensure points are integers for OpenCV
    pts = prediction_seq.astype(int).reshape(-1, 1, 2)

    # Draw the polyline (thin trajectory)
    #result_im = cv2.polylines(result_im, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Draw small circles ("tiny spheres") at each point
    for (x, y) in pts[:, 0]:
        result_im = cv2.circle(result_im, (x, y), radius=2, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return result_im




if __name__ == "__main__":

    # spheres count
    n_objects = 5

    # time step
    dt = 0.01

    prediction_steps = 32

    q = 0.1
    r = 10**-4

    # image dims
    width = 512
    height = 512

    # dynamical system model
    bs = moving_objects.MovingSpheres(n_objects, dt)
    #bs = ColidingSpheres(n_objects, dt)

    colors = numpy.random.rand(n_objects, 3)    

    
    kalman_filter = tracking_kalman_filter.TrackingKalmanFilter(q, r, 2, n_objects, dt)
    

    while True:
        x_gt_pos = bs.step()

        x_obs_pos = x_gt_pos + r*numpy.random.randn(n_objects, 2)

        pred = kalman_filter.step(x_obs_pos)
        prediction_seq = kalman_filter.prediction(prediction_steps)

        
        result_im = numpy.zeros((height, width, 3), dtype=numpy.float32)

        prediction_seq[:, :, 0] = 0.5*(prediction_seq[:, :, 0] + 1.0)*width
        prediction_seq[:, :, 1] = 0.5*(prediction_seq[:, :, 1] + 1.0)*height

        for i in range(n_objects):  

            x = int((x_gt_pos[i][0] + 1.0)*0.5*width)
            y = int((x_gt_pos[i][1] + 1.0)*0.5*height) 
            c = colors[i]
            
            cv2.circle(result_im, (x, y), 10, (1, 1, 1), -1)
            cv2.circle(result_im, (x, y), 8, c, -1) 


            result_im = plot_trajectory(result_im, prediction_seq[:, i], (0, 1, 0))



        result_im = cv2.putText(result_im, "current position", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (1, 1, 1), 2, cv2.LINE_AA) 
        result_im = cv2.putText(result_im, "kalman  prediction", (10, 40), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 1, 0), 2, cv2.LINE_AA) 

        cv2.imshow("visualisation", result_im)
        cv2.waitKey(1)

    

    '''
    x(n+1) = x(n) + v(n)dt
    v(n+1) = v(n)
    '''
    mat_a = numpy.array([[1.0, dt], [0.0, 1.0]])
    mat_b = numpy.zeros((2, 1))
    mat_q = numpy.eye(2)

    # observation matrix, only position is observed
    mat_h = numpy.array([[1.0, 0.0]])

    mat_r = numpy.ones((1, ))


    kf = kalman_filter.KalmanFilter(mat_a, mat_b, mat_q, mat_r, mat_h)