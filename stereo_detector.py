import cv2
import numpy as np
import bbox_visualizer as bbv
from detector import YOLO
from util import DLT, get_projection_matrix
from detect import parse_calibration_settings_file

def main(input_stream0, input_stream1, P0, P1):
	yolo = YOLO()

	cap0 = cv2.VideoCapture(input_stream0)
	# cap1 = cv2.VideoCapture(input_stream1)
	cv2.namedWindow('YOLO Stereo', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

	centroid0, centroid1 = (0,0), (0,0)
	p3d = []
 
	while True:
		ret0, frame0 = cap0.read()
		# ret1, frame1 = cap1.read()
		ret1, frame1 = ret0, frame0
		
		if not (ret0 and ret1):
			break
		
		ball_bboxes_0, _, _ = yolo.find(frame0)
		ball_bboxes_1, _, _ = yolo.find(frame1)
		
		if ball_bboxes_0 and ball_bboxes_1:
			print(ball_bboxes_0)
			frame0[:] = bbv.draw_rectangle(frame0, ball_bboxes_0, (0,255,0))
			x_min, y_min, x_max, y_max = ball_bboxes_0
			centroid0 = ((x_min + x_max)//2, (y_min + y_max)//2)
				
			frame1[:] = bbv.draw_rectangle(frame1, ball_bboxes_1, (0,255,0))
			x_min, y_min, x_max, y_max = ball_bboxes_1
			centroid1 = ((x_min + x_max)//2, (y_min + y_max)//2)
	
			#calculate 3d position
			p3d = DLT(P0, P1, centroid0, centroid1) #calculate 3d position of keypoint
				
			print(f"Selected coordinates: {centroid0}, {centroid1}")
			print(f"3D coordinates: {p3d}\n")
			
		img = np.concatenate((frame0, frame1), axis=1)
  
		cv2.putText(img, f"Selected coordinates: {centroid0}, {centroid1}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)         
		cv2.putText(img, f"3D coordinates: {p3d}",(50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2) 
		
		cv2.imshow('YOLO Stereo', img)

		k = cv2.waitKey(1)
		if k == ord('q'):
			break
			
if __name__ == "__main__":
	calibration_settings = parse_calibration_settings_file("./calibration_settings.yaml")

	input_stream1 = calibration_settings["camera0"]
	input_stream2 = calibration_settings["camera1"]

	print(input_stream1, input_stream2)
	#get projection matrices
	P0 = get_projection_matrix(0)
	P1 = get_projection_matrix(1)

	# kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)
	main(input_stream1, input_stream2, P0, P1)
			