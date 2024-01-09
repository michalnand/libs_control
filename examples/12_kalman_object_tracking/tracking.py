import LibsControl
import numpy
import matplotlib.pyplot as plt

import cv2

if __name__ == "__main__":

    source= cv2.VideoCapture(0)


    while True:
        ret, frame = source.read()

        frame = numpy.array(frame/255.0).astype(numpy.float32)

        frame_denoised = cv2.GaussianBlur(frame, (7, 7), 1)


        #frame_norm = frame_denoised/numpy.expand_dims(frame_denoised.sum(axis=2), 2)

        hsv = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2HSV)


        '''
        row = numpy.argmax(numpy.max(frame_denoised[:, :, 0], axis=1))
        col = numpy.argmax(numpy.max(frame_denoised[:, :, 0], axis=0))
    
        cv2.circle(frame, (col, row), 20, (0, 1, 0), -1)
        '''


        cv2.imshow('frame', hsv[:, :, 0])
        if cv2.waitKey(1) == ord('q'):
            break
