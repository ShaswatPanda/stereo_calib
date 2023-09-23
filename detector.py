import torch
import numpy as np
import matplotlib.pyplot as plt

class YOLO:
    def __init__(self):
        self.yolo = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # self.yolo = torch.hub.load('./yolov5', 'custom', path='./weights/GeneralisedBest.pt', source='local')
        # self.yolo = torch.hub.load('./yolov5', 'custom', path='./weights/Generalisedv5S.pt', source='local')
        # self.yolo = torch.hub.load('./yolov5', 'custom', path='./weights/beterFullbestN.pt', source='local')
        self.yolo.conf = 0.4
        if torch.cuda.is_available():
            self.yolo.to('cuda:0')
        
    def find(self, frame):
        preds = self.yolo(frame, size=640)
        # print(preds)
        box, conf = self.parse_predictions(preds)
        # call human detection 
        person_centroids, person_bboxes = self.parse_predictions_person(preds)
        return box, conf, person_bboxes
    
    def find_cropped(self, frame, margin_percent = .3):
        h_orig,w_orig = frame.shape[:2]
        # print("frame shape", frame.shape[:2])
        left, top, right, bottom = int(margin_percent*w_orig), int(margin_percent*h_orig), w_orig-int(margin_percent*w_orig), h_orig-int(margin_percent*h_orig)
        h_crop, w_crop = bottom-top, right-left 
        detection_frame = frame[top:bottom, left:right].copy()  
        
        # plt.imshow(detection_frame)
        # plt.title("Detection frame")
        # # print("Detection frame")
        # plt.show()
        preds = self.yolo(detection_frame, size=w_crop)
        box, conf = self.parse_predictions(preds)
        if box:
            xmin, ymin, xmax, ymax = box
            box = [xmin+left, ymin+top, xmax+left, ymax+top]
            
        # call human preds
        person_centroids, person_bboxes = self.parse_predictions_person(preds)
        # normalisation of coords
        if person_bboxes:
            for box in person_bboxes:
                xmin, ymin, xmax, ymax = box
                box = [xmin+left, ymin+top, xmax+left, ymax+top]
        if person_centroids:
            for centroid in person_centroids:
                x, y = centroid
                centroid = [x+left, y+top]

        # To find the detection region in a frame
        # return box, conf, detection_frame, [(left, top), (right, bottom)]
        return box, conf, detection_frame, person_bboxes

    def parse_predictions(self, preds):
        preds = preds.pandas().xyxy[0]
        ball_preds =  preds[preds['name'] == 'ball']  # Change class0 to ball for non-tensorrt model
        
        if len(ball_preds)>0:
            preds = np.array(ball_preds.to_records(index=False).tolist())
            # print(preds)
            conf = float(preds[0][4])
            # Choosing the most confident prediction
            idx = np.argsort(preds[:,4])[-1]
            ball_box = preds[idx, :4].astype(float).astype(int).tolist()
            return ball_box, conf
        
        return [], 0
    
    # function to parse predictions for person class and return centroids and bboxes
    def parse_predictions_person(self, preds):
        preds = preds.pandas().xyxy[0]
        person_preds =  preds[preds['name'] == 'person']  # Change class1 to person for non-tensorrt model
        preds = np.array(person_preds.to_records(index=False).tolist())
        person_bboxes = []
        centroids = []
        for i in preds:
            bbox = i[:4].astype(float).astype(int).tolist()
            x_min, y_min, x_max, y_max = bbox
            person_bboxes.append(bbox)
            centroids.append(((x_min + x_max)//2, (y_min + y_max)//2))
        return centroids, person_bboxes
