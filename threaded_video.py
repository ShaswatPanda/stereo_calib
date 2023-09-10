import cv2
import time
from threading import Thread

def single_threaded_video_stream():
    vcap = cv2.VideoCapture(0)
    if vcap.isOpened() is False :
        print("[Exiting]: Error accessing webcam stream.")
        exit(0)
    fps_input_stream = int(vcap.get(5)) # get fps of the hardware
    print("FPS of input stream: {}".format(fps_input_stream))
    grabbed, frame = vcap.read() # reading single frame for initialization/ hardware warm-up

    # processing frames in input stream
    num_frames_processed = 0 
    start = time.time()
    while True :
        grabbed, frame = vcap.read()
        if grabbed is False :
            print('[Exiting] No more frames to read')
            break
        
        num_frames_processed += 1
        # displaying frame 
        cv2.imshow('frame' , frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()

    # printing time elapsed and fps 
    elapsed = end-start
    fps = num_frames_processed/elapsed 
    print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
    # releasing input stream , closing all windows 
    vcap.release()
    cv2.destroyAllWindows()

# defining a helper class for implementing multi-threading 
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
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
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()
    # method to return latest read frame 
    def read(self):
        return self.frame
    # method to stop reading frames 
    def stop(self):
        self.stopped = True
     
def multi_threaded_video_stream():
    # initializing and starting multi-threaded webcam input stream 
    webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
    webcam_stream.start()
    # processing frames in input stream
    num_frames_processed = 0 
    start = time.time()
    while True :
        if webcam_stream.stopped is True :
            break
        else :
            frame = webcam_stream.read()
            num_frames_processed += 1
            
        # displaying frame 
        cv2.imshow('frame' , frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()
    webcam_stream.stop() # stop the webcam stream

    # printing time elapsed and fps 
    elapsed = end-start
    fps = num_frames_processed/elapsed 
    print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
    # closing all windows 
    cv2.destroyAllWindows()
    
def main():
    print(f"Single-threaded stream...")
    single_threaded_video_stream()
    print(f"Multi-threaded stream...")
    multi_threaded_video_stream()
    
if __name__ == "__main__":
    main()