import cv2
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import requests
import imutils
import atexit
import time
from threading import Thread

ip_webcam = False
#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True

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
    
    def start(self):
        self.thread.start()
     
    def stop(self):
        print("Stopping stream...")
        self.capture.release()
        
# defining a helper class for implementing multi-threading 
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.capture = cv2.VideoCapture(self.stream_id)
        if self.capture.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.capture.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from capture stream for initializing 
        self.grabbed , self.frame = self.capture.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.capture.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.capture.release()
    # method to return latest read frame 
    def grab_frame(self):
        return self.grabbed, self.frame
    # method to stop reading frames 
    def stop(self):
        print(f"Stopping stream {self.stream_id}...")
        self.stopped = True

class GetCheckerboard():
    def __init__(self, rows, columns) -> None:
        self.rows = rows
        self.columns = columns
        self.frame_stack = []
        self.frame = None
        
        self.stopped = True
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        pass
    
    def start(self):
        self.stopped = False
        self.t.start()
    
    def update(self):
        while True :
            if self.stopped is True :
                break
            if self.grabbed is False :
                self.stopped = True
                break
            if self.frame_stack:
                frame = self.frame_stack.pop()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (self.rows, self.columns), None)
                conv_size = (11, 11)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                if ret == True:
                    corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
                    self.frame = cv2.drawChessboardCorners(frame, (self.rows,self.columns), corners, ret)
                
    def get_checkerboard(self, frame):
        self.frame_stack.append(frame)
        return self.frame

    def stop(self):
        self.stopped = True

#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


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


#Open camera stream and save frames
def save_frames_single_camera(camera_name):

    #create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    #get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']

    #open video stream and change resolution.
    #Note: if unsupported resolution is used, this does NOT raise an error.
    # cap = cv2.VideoCapture(camera_device_id)
    # cap.set(3, width)
    # cap.set(4, height)

    # streamer = ThreadedCamera(camera_device_id)
    streamer = WebcamStream(camera_device_id)
    atexit.register(streamer.stop)
    streamer.capture.set(3, width)
    streamer.capture.set(4, height)
    streamer.start()
    
    # while streamer.status == False:
    #     continue
    time.sleep(1)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0
    frame_number = 0
    t1 = time.time()
        
    while True:
        ret, frame = streamer.grab_frame()
        if ret == False:
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv2.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)
        
        # gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        # conv_size = (11, 11)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # if ret == True:
        #     corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
        #     cv2.drawChessboardCorners(frame_small, (rows,columns), corners, ret)

        if not start:
            cv2.putText(frame_small, "Press SPACEBAR to start collection frames", (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
        
        if start:
            cooldown -= 1
            cv2.putText(frame_small, "Cooldown: " + str(cooldown), (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(frame_small, "Num frames: " + str(saved_count), (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time
        
        frame_number += 1
        if frame_number % 100 == 0:
            t2 = time.time()
            time_taken = t2 - t1
            print(f"Real-time FPS: {100/time_taken}")
            t1 = t2
        cv2.imshow('frame_small', frame_small)
        k = cv2.waitKey(1)
        
        if k == 27: # ESC
            quit()
        if k == ord('q'): # Q
            quit()     
        if k == 32:
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break
    streamer.stop()
    cv2.destroyAllWindows()


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv2.imread(imname) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv2.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

            cv2.imshow('img', frame)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv2.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


#open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):

    #create frames directory
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    #settings for taking data
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']

    #open the video streams
    # cap0 = cv2.VideoCapture(calibration_settings[camera0_name])
    # if not ip_webcam:
    #     cap1 = cv2.VideoCapture(calibration_settings[camera1_name])
        
    # streamer_left = ThreadedCamera(calibration_settings[camera0_name])
    streamer_left = WebcamStream(calibration_settings[camera0_name])
    atexit.register(streamer_left.stop)
    # streamer_right = ThreadedCamera(calibration_settings[camera1_name])
    streamer_right = WebcamStream(calibration_settings[camera1_name])
    atexit.register(streamer_right.stop)

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    # cap0.set(3, width)
    # cap0.set(4, height)
    # if not ip_webcam:
    #     cap1.set(3, width)
    #     cap1.set(4, height)
    
    streamer_left.capture.set(3, width)
    streamer_left.capture.set(4, height)
    streamer_right.capture.set(3, width)
    streamer_right.capture.set(4, height)
    
    streamer_left.start()
    streamer_right.start()

    # cv2.namedWindow("frame0_small", cv2.WINDOW_NORMAL) 
    # cv2.namedWindow("frame1_small", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cooldown = cooldown_time
    start = False
    saved_count = 0
    frame_number = 0
    t1 = time.time()
    
    while True:

        # ret0, frame0 = cap0.read()
        # if not ip_webcam:
        #     ret1, frame1 = cap1.read()
        # else:
        #     url = "http://192.168.1.2:8080/shot.jpg"
        #     img_resp = requests.get(url)
        #     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        #     img = cv2.imdecode(img_arr, -1)
        #     ret1, frame1 = True, img

        ret0, frame0 = streamer_left.grab_frame()
        ret1, frame1 = streamer_right.grab_frame()
        
        if not (ret0 and ret1):
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv2.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv2.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)
        
        # # REAL-TIME CHECKERBOARD PLOT
        # gray0 = cv2.cvtColor(frame0_small, cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray0, (rows, columns), None)
        # conv_size = (11, 11)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # if ret == True:
        #     corners = cv2.cornerSubPix(gray0, corners, conv_size, (-1, -1), criteria)
        #     cv2.drawChessboardCorners(frame0_small, (rows,columns), corners, ret)
            
        # gray1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray1, (rows, columns), None)
        # conv_size = (11, 11)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # if ret == True:
        #     corners = cv2.cornerSubPix(gray1, corners, conv_size, (-1, -1), criteria)
        #     cv2.drawChessboardCorners(frame1_small, (rows,columns), corners, ret)

        if not start:
            cv2.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(frame0_small, "Press SPACEBAR to start collection frames", (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
        
        if start:
            cooldown -= 1
            cv2.putText(frame0_small, "Cooldown: " + str(cooldown), (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(frame0_small, "Num frames: " + str(saved_count), (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            
            cv2.putText(frame1_small, "Cooldown: " + str(cooldown), (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(frame1_small, "Num frames: " + str(saved_count), (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame0)

                savename = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        combined_frame0_frame_1 = np.concatenate((frame0_small, cv2.resize(frame1_small, (frame0_small.shape[1], frame0_small.shape[0]))))
        cv2.imshow('img', combined_frame0_frame_1)
        
        frame_number += 1
        if frame_number % 100 == 0:
            t2 = time.time()
            time_taken = t2 - t1
            print(f"Real-time FPS: {100/time_taken}")
            t1 = t2
            
        k = cv2.waitKey(1)
        if k == 27:
            quit()  
        if k == ord('q'): # Q
            quit()
        if k == 32:
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    streamer_left.stop()
    streamer_right.stop()
    cv2.destroyAllWindows()

#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv2.imread(imname, 2) for imname in c0_images_names]
    c1_images = [cv2.imread(imname, 2) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv2.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            # cv2.imshow('img', frame0)

            cv2.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            # cv2.imshow('img2', frame1)
            
            print(f"Frame 0: {frame0.shape}\tFrame1: {frame1.shape}")
            combined_frame0_frame_1 = np.concatenate((frame0, cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))))
            cv2.imshow('img', combined_frame0_frame_1)
            
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv2.destroyAllWindows()
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv2.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv2.VideoCapture(calibration_settings[camera0_name])
    if not ip_webcam:
        cap1 = cv2.VideoCapture(calibration_settings[camera1_name])

    # cv2.namedWindow("frame0", cv2.WINDOW_NORMAL) 
    # cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    
    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    if not ip_webcam:
        cap1.set(3, width)
        cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        if not ip_webcam:
            ret1, frame1 = cap1.read()
        else:
            url = "http://192.168.1.2:8080/shot.jpg"
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            ret1, frame1 = True, img

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame1, origin, _p, col, 2)

        # cv2.imshow('frame0', frame0)
        # cv2.imshow('frame1', frame1)
        # print(f"Frame 0: {frame0.shape}\tFrame1: {frame1.shape}")
        combined_frame0_frame_1 = np.concatenate((frame0, cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))))
        cv2.imshow('img', combined_frame0_frame_1)

        k = cv2.waitKey(1)
        if k == 27: break

    cv2.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv2.imread(img_path, 2)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv2.putText(frame, "If you don't see detected points, try with a different image", (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow('img', frame)
    cv2.waitKey(0)

    ret, rvec, tvec = cv2.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv2.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv2.imread(image_path0, 2)
    frame1 = cv2.imread(image_path1, 2)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv2.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame1, origin, _p, col, 2)

    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    cv2.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()
    
    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])


    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0') #save frames for camera0
    save_frames_single_camera('camera1') #save frames for camera1


    # """Step2. Obtain camera intrinsic matrices and save them"""
    # #camera0 intrinsics
    # images_prefix = os.path.join('frames', 'camera0*')
    # cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix) 
    # save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    # #camera1 intrinsics
    # images_prefix = os.path.join('frames', 'camera1*')
    # cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    # save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 2))

    save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)


    """Optional. Define a different origin point and save the calibration data"""
    # #get the world to camera0 rotation and translation
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join('frames_pair', 'camera0_4.png'))
    # #get rotation and translation from world directly to camera1
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join('frames_pair', 'camera0_4.png'),
    #                                           os.path.join('frames_pair', 'camera1_4.png'),)

    # #save rotation and translation parameters to disk
    # save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix = 'world_to_') #this will write R and T to disk

