import cv2
import numpy as np
import sys
import requests
import os
import yaml
import atexit
import time
from utils import DLT, get_projection_matrix, write_keypoints_to_disk
from threaded_video import WebcamStreamSynced

ip_cam = False
sync = False
frame_shape = [1280, 720]
global coords

#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    global calibration_settings
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

def run_mp(input_stream1, input_stream2, P0, P1):
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    #input video stream
    #set camera resolutions
    if sync:
        streamer_left = WebcamStreamSynced(input_stream1)
        atexit.register(streamer_left.stop)
        # streamer_right = ThreadedCamera(calibration_settings[camera1_name])
        streamer_right = WebcamStreamSynced(input_stream2)
        atexit.register(streamer_right.stop)
        
        streamer_left.capture.set(3, width)
        streamer_left.capture.set(4, height)
        streamer_right.capture.set(3, width)
        streamer_right.capture.set(4, height)
    
        streamer_left.start()
        streamer_right.start()
        time.sleep(1)
    else:
        cap0 = cv2.VideoCapture(input_stream1)
        if ip_cam:
            caps = [cap0]
        else:
            cap1 = cv2.VideoCapture(input_stream2)
            caps = [cap0, cap1]
        # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
        for cap in caps:
            cap.set(3, frame_shape[1])
            cap.set(4, frame_shape[0])
    

    #containers for detected keypoints for each camera. These are filled at each frame.
    #This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    
    start = False
    cooldown_time = 100
    cooldown = cooldown_time

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    frame_number = 0
    t1 = time.time()
    fps = None
    
    
    while True:

        #read frames from stream
        if sync:
            ret0, frame0 = streamer_left.grab_frame(frame_number=frame_number)
            ret1, frame1 = streamer_right.grab_frame(frame_number=frame_number)
        else:
            ret0, frame0 = cap0.read()
            if not ip_cam:
                ret1, frame1 = cap1.read()
            else:
                url = "http://192.168.1.2:8080/shot.jpg"
                img_resp = requests.get(url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)
                ret1, frame1 = True, img


        if not ret0 or not ret1: 
            print('Cameras not returning video data. Exiting...')
            break

        if not sync:
            """
            crop to 720x720.
            Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
            """
            if frame0.shape[1] != 720:
                frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
                frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            
            if ip_cam: 
                frame1 = frame1[frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2, :, :]

        combined_frame0_frame_1 = np.concatenate((frame0, cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))), axis=1)

        if not start:
            cv2.putText(combined_frame0_frame_1, "Make sure both cameras can see the calibration pattern well", (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(combined_frame0_frame_1, "Press SPACEBAR to start collection frames", (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
        
        if start:
            cooldown -= 1
            cv2.putText(combined_frame0_frame_1, "Cooldown: " + str(cooldown), (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            
            coords = []            
            if cooldown <= 0:                
                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if x < img.shape[1] // 2:
                            coords.append((x,y))
                        else:
                            coords.append((x - img.shape[1] // 2,y))
                                                        
                        cv2.putText(img, f'({coords[-1][0]},{coords[-1][0]})',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2) 
                        cv2.circle(img, (x,y), 5, (0,0,255), -1)
                        cv2.imshow('Select Coordinates', img)
                
                cv2.imshow('Select Coordinates', img)
                cv2.setMouseCallback('Select Coordinates', click_event)
                cv2.waitKey(0)
                cv2.destroyWindow('Select Coordinates')
                
                ball_coords_frame_0 = coords[0]
                ball_coords_frame_1 = coords[1]

                #calculate 3d position
                if ball_coords_frame_0[0] == -1 or ball_coords_frame_1[0] == -1:
                    p3d = [-1, -1, -1]
                else:
                    p3d = DLT(P0, P1, ball_coords_frame_0, ball_coords_frame_1) #calculate 3d position of keypoint
                    
                print(f"Selected coordinates: {coords}")
                print(f"3D coordinates: {p3d}\n")
        
                cooldown = cooldown_time
                start = False
                pass

        # # cv2.imshow('cam1', frame1)
        # # cv2.imshow('cam0', frame0)
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

        k = cv2.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.
        if k == ord('q'): break #q
        if k == 32:
            #Press spacebar to start data collection
            start = True
    
    if sync:
        streamer_left.stop()
        streamer_right.stop()   
    else:
        for cap in caps:
            cap.release()
    cv2.destroyAllWindows()


    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    parse_calibration_settings_file("./calibration_settings.yaml")
    #this will load the sample videos if no camera ID is given
    input_stream1 = calibration_settings["camera0"]
    input_stream2 = calibration_settings["camera1"]
    # input_stream1 = "./videos/left_stream.avi"
    # input_stream2 = "./videos/right_stream.avi"
    
    print(input_stream1, input_stream2)
    #get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('./camera_parameters/kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('./camera_parameters/kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('./camera_parameters/kpts_3d.dat', kpts_3d)