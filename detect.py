import cv2
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk
import requests

frame_shape = [720, 1280]
ip_cam = False
global coords

def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv2.VideoCapture(input_stream1)
    if ip_cam:
        caps = [cap0]
    else:
        cap1 = cv2.VideoCapture(input_stream2)
        caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
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
    
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        if not ip_cam:
            ret1, frame1 = cap1.read()
        else:
            url = "http://192.168.1.2:8080/shot.jpg"
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            ret1, frame1 = True, img

        if not ret0 or not ret1: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        
        if ip_cam: 
            frame1 = frame1[frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2, :, :]

        # print(frame0.shape, frame1.shape)
        img = np.concatenate((frame0, frame1), axis=1)


        if not start:
            cv2.putText(img, "Press SPACEBAR to start collection frames", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv2.putText(img, "Cooldown: " + str(cooldown), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
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
        cv2.imshow('img', img)

        k = cv2.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.
        if k == ord('q'): break #q
        if k == 32:
            #Press spacebar to start data collection
            start = True


    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()


    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    #this will load the sample videos if no camera ID is given
    input_stream1 = './videos/left_stream.avi'
    input_stream2 = './videos/right_stream.avi'
    # input_stream1 = 0
    # input_stream2 = 1
    
    #get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('./camera_parameters/kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('./camera_parameters/kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('./camera_parameters/kpts_3d.dat', kpts_3d)