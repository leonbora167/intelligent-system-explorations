import argparse 
import cv2
import torch 
from sort import Sort
import numpy as np
import csv 

parser = argparse.ArgumentParser(description="Vehicle Tracking Only \n Enter the 'input' path")
parser.add_argument("--input", type=str, required=True, help="path to video file or rtsp stream")
parser.add_argument("--output", type=str, required=True, help="path to output file for csv")
args = parser.parse_args()

#Creating a CSV File to save the detections
outputfile = args.output
csv_file = open(outputfile, "w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Class", "Confidence", "Coordinates", "Vehicle_ID"])

#Yolo Model Loading
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5n.pt")

#Load SORT
tracker = Sort()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Video / Stream Source
source = args.input

#Video Stream
vid_cap = cv2.VideoCapture(source)
frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_id = 0

while True:
    ret, frame = vid_cap.read()
    if not ret:
        break 

    #Yolo inference
    results = model(frame)
    detected_vehicles = results.pandas().xyxy[0]
    bboxes = results.pandas().xyxy[0]
    vehicles = bboxes[bboxes['name'].isin(['car', 'truck', 'bus', 'bike'])] #Get the rows of the dataframe where vehicles have been detected
    
    #SORT Tracking preparing the yolo detections
    detections = []
    for _, row in vehicles.iterrows():
        x1, y1, x2, y2, conf, v_class = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']]
        detection = [x1, y1, x2, y2, conf]
        detections.append(detection)

    #Getting the updates from SORT
    tracked_objects = tracker.update(np.array((detections))) #Returns coordinates of the tracked objects with the "ID"
    
    # Draw bounding boxes and IDs on the frame
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        coords = str(x1)+","+str(y1)+","+str(x2)+","+str(y1)
        csv_writer.writerow([frame_id, v_class, conf, coords, obj_id])

    cv2.imshow("Window", frame)
    if(cv2.waitKey(25) & 0xFF == ord("q")):
        break 

    frame_id += 1

vid_cap.release()
cv2.destroyAllWindows()
    