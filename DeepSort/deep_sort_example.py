import numpy as np
import cv2
from ultralytics import YOLO
import os
import sys
#from deep_sort_realtime.deepsort_tracker import DeepSort
sys.path.append( '/home/prashant/Documents/legion/personal_projects/soccer/deep_sort/')

import dsort_tracker
print("dsort_tracker module path:", dsort_tracker.__file__)
from dsort_tracker import Tracking


# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video source
src = "./D0_shadow.mp4"
cap = cv2.VideoCapture(src)

# Check if the video source is opened
if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_file = "./home_foot_out.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
output_resolution = (640, 480)  # Match the resized frame resolution
out = cv2.VideoWriter(output_file, fourcc, fps, output_resolution)

track_thresh=0.6
track_buffer=100
match_thresh=0.9
frame_rate=fps
#tracker =BYTETracker(track_thresh, track_buffer, match_thresh, frame_rate)
#Initialise the object tracker clas
pth='/home/prashant/Documents/legion/personal_projects/soccer/deep_sort/model_data/mars-small128.pb'
tracker = Tracking(pth)
def draw_track_frame(track_detection,detection_frame):
    for track in track_detection:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            bbox = ltrb

            cv2.rectangle(detection_frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
            cv2.putText(detection_frame, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

       

            cv2.putText(detection_frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    return detection_frame
def draw_track_frame_dsort(track_detection,detection_frame):
    if track_detection:
        for item in track_detection:
                track=item
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                cv2.rectangle(detection_frame, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255), 3)
                cv2.putText(detection_frame, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    return detection_frame   
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or cannot fetch frame.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Perform YOLO detection
    results = model(frame)

    detections = results[0]  # First result for the frame


    # Filter detections for 'person' class (class index 0 in COCO dataset)
    person_detections = detections.boxes[detections.boxes.cls == 0]
    #print("person detection is: ",person_detections)
    
    detections_deepsort = []
    # Get bounding box annotations (x1, y1, x2, y2)
    raw_detection = np.empty((0,6), float)
    for box in person_detections:
        #print("the box are: ",box)
        class_id='0'
        if box.conf.cpu().numpy()>0.4:
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
           
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            confidence=box.conf.cpu().numpy()
            coord=[x1, y1, x2, y2, confidence]
            detections_deepsort.append(coord)

    
    tracks_dsort = tracker.update(frame,detections_deepsort )
   
    
     
    print("deepsort OUTOUT------------------------------------------: ",tracks_dsort)
    # Write the annotated frame to the output video
    frame=draw_track_frame_dsort(tracks_dsort,frame)
    out.write(frame)

    # Press 'q' to exit the loop early (optional for debugging)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
print(f"Video saved to {output_file}")