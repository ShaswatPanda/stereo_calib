import cv2
from threading import Thread
import atexit
import numpy as np  
import os
import yaml

def parse_calibration_settings_file(filename):
    calibration_settings = None
    
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()
        
    return calibration_settings

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            else:
                break
        self.stop()

    def grab_frame(self):
        return self.status, self.frame
     
    def stop(self):
        print("Stopping stream...")
        self.capture.release()
        

def main(args):  
    # stream_link = "rtsp://192.168.100.88:554/1"

    stream_link_left = args["camera0"]
    streamer_left = ThreadedCamera(stream_link_left)
    atexit.register(streamer_left.stop)

    stream_link_right = args["camera1"]
    streamer_right = ThreadedCamera(stream_link_right)
    atexit.register(streamer_right.stop)


    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    
    while(True):  
        ret0, frame0 = streamer_left.grab_frame()
        ret1, frame1 = streamer_right.grab_frame()
        view_resize= 1

        if not (ret0 and ret1):
            break

        frame0_small = cv2.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv2.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)
        

        combined_frame0_frame_1 = np.concatenate((frame0_small, cv2.resize(frame1_small, (frame0_small.shape[1], frame0_small.shape[0]))))
        cv2.imshow('img', combined_frame0_frame_1)

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
 
    cv2.destroyAllWindows()  
    
if __name__ == "__main__":
    args = parse_calibration_settings_file("./calibration_settings.yaml")
    main(args)