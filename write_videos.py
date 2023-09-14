import cv2
from threaded_video import WebcamStreamSynced 
import atexit
import numpy as np  
import os
import yaml
import time
from datetime import datetime

sync = False

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
        

def main(args):  
    result_path = "./videos/results"
    os.makedirs(result_path, exist_ok=True)
    
    if sync: 
        streamer_left = WebcamStreamSynced(args["camera0"])
        atexit.register(streamer_left.stop)
        streamer_right = WebcamStreamSynced(args["camera1"])
        atexit.register(streamer_right.stop)
        
        width_left, height_left = streamer_left.frame.shape[1],streamer_left.frame.shape[0]
        width_right, height_right = streamer_right.frame.shape[1],streamer_right.frame.shape[0]
        
        streamer_left.start()
        streamer_right.start()
        time.sleep(1)
    else:
        streamer_left = cv2.VideoCapture(args["camera0"])
        streamer_right = cv2.VideoCapture(args["camera1"])
        
        width_left  = int(streamer_left.get(cv2.CAP_PROP_FRAME_WIDTH))   
        height_left = int(streamer_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_right  = int(streamer_right.get(cv2.CAP_PROP_FRAME_WIDTH))   
        height_right = int(streamer_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print((width_left, width_right), (width_right, height_right))
    # assert (width_left, width_right) == (width_right, height_right), "PLEASE ENSURE THE SIZES OF THE STREAMS ARE THE SAME" 
    
    run_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
    result_left = cv2.VideoWriter(f'{result_path}/left_stream_{run_time}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width_left, height_left))
    result_right = cv2.VideoWriter(f'{result_path}/right_stream_{run_time}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width_right, height_right))

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    frame_number = 0
    t1 = time.time()
    fps = None

    
    while(True):  
        
        view_resize= 1
        if sync:
            ret0, frame0 = streamer_left.grab_frame(frame_number=frame_number)
            ret1, frame1 = streamer_right.grab_frame(frame_number=frame_number)
        else:
            ret0, frame0 = streamer_left.read()
            ret1, frame1 = streamer_right.read()

        if not (ret0 and ret1):
            break

        frame0_small = cv2.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv2.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)
        
        write_frame0, write_frame1 = frame0_small, cv2.resize(frame1_small, (frame0_small.shape[1], frame0_small.shape[0]))
        result_left.write(write_frame0)
        result_right.write(write_frame1)

        combined_frame0_frame_1 = np.concatenate((frame0_small, cv2.resize(frame1_small, (frame0_small.shape[1], frame0_small.shape[0]))), axis=1)
        if sync:
            cv2.putText(combined_frame0_frame_1, f"FRAME - 0:{streamer_left.frame_number}    1:{streamer_right.frame_number}    Delta: {streamer_left.frame_number - streamer_right.frame_number}", (30,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(combined_frame0_frame_1, f"FRAME NUMBER: {frame_number}", (30,110), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(combined_frame0_frame_1, f"FPS: {fps}", (30,140), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('img', combined_frame0_frame_1)

        frame_number += 1
        if frame_number % 100 == 0:
            t2 = time.time()
            time_taken = t2 - t1
            fps = round(100/time_taken, 2)
            print(f"Real-time FPS: {fps}")
            t1 = t2
            
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
  
    result_left.release()
    result_right.release()
    streamer_left.stop()
    streamer_right.stop()
    cv2.destroyAllWindows()  
    
if __name__ == "__main__":
    args = parse_calibration_settings_file("./calibration_settings.yaml")
    main(args)