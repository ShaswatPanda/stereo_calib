import cv2
import numpy as np
import sys
import requests
import os
import yaml
import atexit
import time
from util import DLT, get_projection_matrix, write_keypoints_to_disk
from threaded_video import WebcamStreamSynced

video = True
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
    return calibration_settings

def run_mp(input_stream1, input_stream2, P0, P1):
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    #input video stream
    #set camera resolutions
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

    cooldown_time = 100
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    frame_number = 0
    take_coords = True

    if video:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break
            
            combined_frame0_frame_1 = np.concatenate((frame0, cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))), axis=1)
        
            if take_coords:        
                coords = []            
                    
                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        (x_old, y_old) = (x,y)
                        if not x < combined_frame0_frame_1.shape[1] // 2:
                            (x,y) = (x - combined_frame0_frame_1.shape[1] // 2,y)
                        coords.append((x,y))
                        
                        cv2.circle(combined_frame0_frame_1, (x_old,y_old), 5, (0,0,255), -1)
                        cv2.putText(combined_frame0_frame_1, f"({x},{y})",(x_old,y_old), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                  
                        cv2.imshow('Select Coordinates', combined_frame0_frame_1)
                        
                cv2.namedWindow('Select Coordinates',  cv2.WINDOW_NORMAL)
                cv2.imshow('Select Coordinates', combined_frame0_frame_1)
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
                
                take_coords = False
                pass
        

            cv2.putText(combined_frame0_frame_1, f"Selected coordinates: {coords}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)         
            cv2.putText(combined_frame0_frame_1, f"3D coordinates: {p3d}",(50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2) 
            cv2.imshow('img', combined_frame0_frame_1)
            k = cv2.waitKey(1)
            if k == ord('s'):
                print("Getting coords...")
                take_coords = True
            if k == ord('q'):
                break
            

            

    else:
        ret0, frame0 = True, cv2.imread("../Umbrella-perfect/im0.png")
        ret1, frame1 = True, cv2.imread("../Umbrella-perfect/im1.png")
        
        combined_frame0_frame_1 = np.concatenate((frame0, cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))), axis=1)
        
        if take_coords:        
            coords = []            
                
            def click_event(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    (x_old, y_old) = (x,y)
                    if not x < combined_frame0_frame_1.shape[1] // 2:
                        (x,y) = (x - combined_frame0_frame_1.shape[1] // 2,y)
                    coords.append((x,y))
                    
                    cv2.circle(combined_frame0_frame_1, (x_old,y_old), 15, (0,0,255), -1)
                    cv2.putText(combined_frame0_frame_1, f"({x},{y})",(x_old,y_old), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
                                                
                    cv2.imshow('Select Coordinates', combined_frame0_frame_1)
                    
            cv2.namedWindow('Select Coordinates',  cv2.WINDOW_NORMAL)
            cv2.imshow('Select Coordinates', combined_frame0_frame_1)
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
            
            take_coords = False
            pass
        

        cv2.putText(combined_frame0_frame_1, f"Selected coordinates: {coords}",(150,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)         
        cv2.putText(combined_frame0_frame_1, f"3D coordinates: {p3d}",(150,300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10) 
        cv2.imshow('img', combined_frame0_frame_1)
        k = cv2.waitKey(0)
        if k == ord('s'):
            print("Getting coords...")
            take_coords = True

    # return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


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

    # kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)
    run_mp(input_stream1, input_stream2, P0, P1)

    # #this will create keypoints file in current working folder
    # write_keypoints_to_disk('./camera_parameters/kpts_cam0.dat', kpts_cam0)
    # write_keypoints_to_disk('./camera_parameters/kpts_cam1.dat', kpts_cam1)
    # write_keypoints_to_disk('./camera_parameters/kpts_3d.dat', kpts_3d)