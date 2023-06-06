import argparse 
import cv2
import torch 
from sort import Sort
import numpy as np
from scipy.spatial.distance import euclidean
import csv

parser = argparse.ArgumentParser(description="Vehicle Tracking Only \n Enter the 'input' path")
parser.add_argument("--input", type=str, required=True, help="path to video file or rtsp stream")
parser.add_argument("--output", type=str, required=True, help="path to output file for csv")
parser.add_argument("--output2", type=str, required=True, help="path for csv files to save the safety results")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Yolo Model Loading
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5l.pt",device="0")
model2 = torch.hub.load("ultralytics/yolov5","custom",path="yolov5s_custom.pt",device="0")


#Load SORT
tracker = Sort()


#Creating a CSV File to save the detections
outputfile = args.output
csv_file = open(outputfile, "w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Class", "Confidence", "Coordinates", "Vehicle_ID","Vehicle_Speed"])


#Create a CSV File to save the safety outputs
outputfile2 = args.output2
csv_file2 = open(outputfile2, "w")
csv_writer2 = csv.writer(csv_file2) 
csv_writer2.writerow(["Frame", "Class", "Confidence", "Coordinates", "Vehicle_ID", "Safety_Flag"])


#Video / Stream Source
source = args.input

#Video Stream
vid_cap = cv2.VideoCapture(source)
frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
print("Frame Rate of video is ", frame_rate)
frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def safety_detect(x1,y1,x2,y2, img):
    try:
        results = model2(img[y1:y2, x1:x2])
    except:
        return 0
    #detected_safety = results.pandas().xyxy[0]
    bboxes = results.pandas().xyxy[0]
    safety_flags = bboxes[bboxes["name"].isin(["helmet","no_helmet","seatbelt","no_seatbelt"])]
    for _, row in safety_flags.iterrows():
        x1, y1, x2, y2, conf, safety_class = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']]
        dets = [x1, y1, x2, y2, conf]
        coords2 = str(x1)+","+str(y1)+","+str(x2)+","+str(y1)
        csv_writer.writerow([frame_id, safety_class, conf, coords2, obj_id,speed])



#Refer to the Readme for explanation regarding these values
#ROI Values for Detection in Road
"""
ux1, uy1 = 910,500
ux2, uy2 = 1400,500
lx1, ly1 = 950,1000
lx2, ly2 = 1750, 1000
road_length = 34.5
"""


def calculate_speed(prev_bbox, curr_bbox, time_diff, pixel_to_meter_ratio):
    prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
    curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
    distance = euclidean(prev_center, curr_center) #value is in px
    distance = distance * pixel_to_meter_ratio
    speed = (distance/time_diff) * 3.6 #convert speed from m/s to km/hr
    return speed

prev_frame_time = 0
prev_bboxes = {}
frame_id = 0
pixel_to_meter_ratio = 0.02663 #1px = 0.02663m

while True:
    ret, frame = vid_cap.read()
    if not ret:
        break 
    frame_id = frame_id + 1
    #Yolo inference
    results = model(frame)
    detected_vehicles = results.pandas().xyxy[0]
    bboxes = results.pandas().xyxy[0]
    vehicles = bboxes[bboxes['name'].isin(['car', 'truck', 'bus', 'motorcycle'])] #Get the rows of the dataframe where vehicles have been detected
    
    #SORT Tracking preparing the yolo detections
    detections = []
    for _, row in vehicles.iterrows():
        x1, y1, x2, y2, conf, v_class = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']]
        detection = [x1, y1, x2, y2, conf]
        condition1 = ((int(y2)>=500) and (int(x1)>=910))
        condition2 = (int(x2)<=1400)
        #condition2 = int(x2<lx2) 
        if(condition1 and condition2):
            detections.append(detection)

    #Getting the updates from SORT
    try:
        tracked_objects = tracker.update(np.array((detections))) #Returns coordinates of the tracked objects with the "ID"
    except:
        continue
    
    current_frame_time = frame_id 
    time_diff = (current_frame_time - prev_frame_time) * (1/30)


    # Draw bounding boxes and IDs on the frame
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        bbox = (x1,y1,x2,y2,obj_id)
        safety_detect(int(x1),int(y1),int(x2),int(y2),frame) #Check for safety features
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #Check if vehicle is present in previous frame
        if obj_id in prev_bboxes:
            prev_bbox = prev_bboxes[obj_id]
            speed = calculate_speed(prev_bbox, bbox, time_diff, pixel_to_meter_ratio)
            cv2.putText(frame, f"Speed: {speed:.2f} km/h", (int(x2), int(y1) - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            coords = str(x1)+","+str(y1)+","+str(x2)+","+str(y1)
            csv_writer.writerow([frame_id, v_class, conf, coords, obj_id,speed])
        prev_bboxes[obj_id] = bbox 

    cv2.rectangle(frame, (910,500), (1400,500), (255,255,0),3)
    cv2.rectangle(frame, (950,910), (1600,910), (0,0,255), 3)
    cv2.imshow("Window", frame)
    if(cv2.waitKey(25) & 0xFF == ord("q")):
        break 

    prev_frame_time = current_frame_time

vid_cap.release()
cv2.destroyAllWindows()
    