import cv2  
import numpy as np  
  
cap = cv2.VideoCapture("./stereo-example.mp4")  

# h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

h = 480
w = 1280
fps = 30

left_result = cv2.VideoWriter('left_stream.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, (w//2,h))
right_result = cv2.VideoWriter('right_stream.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, (w//2,h))

while(True):  
    ret, frame = cap.read()  
    if not ret:
        break
    frame_left = frame[:, :frame.shape[1] // 2, :]  
    frame_right = frame[:, frame.shape[1] // 2:, :]  
    # print(frame.shape, frame_left.shape)
    # cv2.imshow('frame_left', frame_left)  
    # cv2.imshow('frame_right', frame_right)
    # Display the resulting frame  
    cv2.imshow('frame', frame)  
    left_result.write(frame_left)
    right_result.write(frame_right)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# When everything done, release the capture  
cap.release()  
left_result.release()
right_result.release()
cv2.destroyAllWindows()  